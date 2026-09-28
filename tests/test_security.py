"""Security and reliability behaviour, exercised through httpx's MockTransport (no network)."""

from __future__ import annotations

import asyncio
import json
import time
import warnings
from typing import Dict, List
from unittest.mock import MagicMock, patch

import httpx
import pytest

import signalvault
from signalvault import (
    AsyncSignalVaultClient,
    SignalVaultBlockedError,
    SignalVaultClient,
    SignalVaultUnavailableError,
    SignalVaultWarning,
)
from signalvault.client import normalize_base_url
from signalvault.tools import sanitize_payload

ALLOW = {"status": 200, "json": {"decision": "allow", "violations": [], "redactions": []}}
MESSAGES = [{"role": "user", "content": "hi"}]


class Server:
    """Records requests; replies from per-event-type queues (default 200 {})."""

    def __init__(self, replies: Dict[str, List[dict]] | None = None, delay: float = 0.0):
        self.replies = replies or {}
        self.delay = delay
        self.requests: List[httpx.Request] = []

    @property
    def bodies(self) -> List[dict]:
        return [json.loads(r.content) for r in self.requests]

    def types(self) -> List[str]:
        return [b["type"] for b in self.bodies]

    def _reply(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queue = self.replies.get(json.loads(request.content)["type"])
        spec = queue.pop(0) if queue else {"status": 200, "json": {}}
        if "text" in spec:
            return httpx.Response(spec["status"], text=spec["text"], headers=spec.get("headers"))
        return httpx.Response(spec["status"], json=spec.get("json", {}), headers=spec.get("headers"))

    def sync_handler(self, request: httpx.Request) -> httpx.Response:
        if self.delay:
            time.sleep(self.delay)
        return self._reply(request)

    async def async_handler(self, request: httpx.Request) -> httpx.Response:
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._reply(request)


def fake_completion():
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "hello"
    resp.usage = MagicMock(prompt_tokens=3, completion_tokens=1)
    return resp


def sync_client(server: Server, **kw) -> SignalVaultClient:
    client = SignalVaultClient(api_key="sk_test_abc", openai_api_key="sk-fake", **kw)
    client._http = httpx.Client(transport=httpx.MockTransport(server.sync_handler))
    client._openai = MagicMock()
    client._openai.chat.completions.create.return_value = fake_completion()
    return client


# ---------------------------------------------------------------------------
# base_url
# ---------------------------------------------------------------------------

class TestBaseUrl:
    def test_accepts_https_and_local_http(self):
        assert normalize_base_url("https://api.signalvault.io/") == "https://api.signalvault.io"
        assert normalize_base_url("http://localhost:4000") == "http://localhost:4000"
        assert normalize_base_url("http://127.0.0.1:4000") == "http://127.0.0.1:4000"
        assert normalize_base_url("http://[::1]:4000") == "http://[::1]:4000"

    def test_refuses_plaintext_to_remote_hosts(self):
        with pytest.raises(ValueError, match="must use https"):
            SignalVaultClient(api_key="k", openai_api_key="x", base_url="http://api.signalvault.io")

    def test_refuses_garbage_and_other_schemes(self):
        for bad in ("api.signalvault.io", "ftp://api.signalvault.io"):
            with pytest.raises(ValueError):
                normalize_base_url(bad)

    def test_rejects_unknown_fail_mode(self):
        with pytest.raises(ValueError, match="fail_mode"):
            SignalVaultClient(api_key="k", openai_api_key="x", fail_mode="maybe")

    def test_http_clients_do_not_follow_redirects(self):
        client = SignalVaultClient(api_key="k", openai_api_key="x")
        assert client._http.follow_redirects is False


# ---------------------------------------------------------------------------
# Headers & event_id
# ---------------------------------------------------------------------------

class TestHeaders:
    def test_accept_user_agent_and_event_id(self):
        server = Server({"ai.request": [ALLOW]})
        client = sync_client(server)
        client.chat.completions.create(model="m", messages=MESSAGES)
        client.flush()

        headers = server.requests[0].headers
        assert headers["accept"] == "application/json"
        assert headers["user-agent"].startswith(f"signalvault-python/{signalvault.__version__} python/")
        response_event = server.bodies[1]
        assert response_event["type"] == "ai.response"
        assert len(response_event["event_id"]) == 36


# ---------------------------------------------------------------------------
# Pre-flight failures
# ---------------------------------------------------------------------------

class TestPreflightFailures:
    @pytest.mark.parametrize("status,body,pattern", [
        (401, {"errors": {"detail": "Unauthorized"}}, "API key"),
        (402, {"error": "subscription_inactive"}, "not active"),
        (403, {"error": "environment_not_allowed"}, "denied access"),
        (429, {"error": "rate_limited"}, "rate limit"),
        (429, {"error": {"type": "trial_limit_exceeded"}}, "trial limit"),
        (503, {}, "API error"),
    ])
    def test_fail_open_warns_without_debug(self, status, body, pattern):
        server = Server({"ai.request": [{"status": status, "json": body}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning, match=pattern):
            client.chat.completions.create(model="m", messages=MESSAGES)
        client._openai.chat.completions.create.assert_called_once()

    def test_warns_once_per_minute_per_cause(self):
        server = Server({"ai.request": [{"status": 401, "json": {}} for _ in range(3)]})
        client = sync_client(server)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                client.chat.completions.create(model="m", messages=MESSAGES)
        assert len([w for w in caught if issubclass(w.category, SignalVaultWarning)]) == 1

    def test_fail_closed_raises_and_skips_provider(self):
        server = Server({"ai.request": [{"status": 401, "json": {}}]})
        client = sync_client(server, fail_mode="closed")
        with pytest.raises(SignalVaultUnavailableError) as exc:
            client.chat.completions.create(model="m", messages=MESSAGES)
        assert exc.value.status == 401
        client._openai.chat.completions.create.assert_not_called()

    def test_non_json_200_is_unavailable_not_a_crash(self):
        server = Server({"ai.request": [{"status": 200, "text": "<html>maintenance</html>"}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning, match="invalid response"):
            client.chat.completions.create(model="m", messages=MESSAGES)
        client._openai.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

class TestDecisions:
    def test_block_raises_typed_error(self):
        violations = [{"rule_id": "r1", "type": "contains_secret", "severity": 9, "action": "block", "details": {}}]
        server = Server({"ai.request": [{"status": 200, "json": {
            "decision": "block", "violations": violations, "redactions": [],
            "dashboard_url": "https://signalvault.io/x",
        }}]})
        client = sync_client(server)
        with pytest.raises(SignalVaultBlockedError) as exc:
            client.chat.completions.create(model="m", messages=MESSAGES)
        err = exc.value
        assert isinstance(err, RuntimeError)
        assert str(err) == "[SignalVault] Request blocked by guardrails (contains_secret)."
        assert err.violations[0].rule_id == "r1"
        assert err.dashboard_url == "https://signalvault.io/x"
        client._openai.chat.completions.create.assert_not_called()

    def test_redactions_array_and_request_unmodified(self):
        server = Server({"ai.request": [{"status": 200, "json": {
            "decision": "redact", "violations": [], "redactions": [{"type": "contains_pii", "count": 1}],
        }}]})
        client = sync_client(server)
        client.chat.completions.create(model="m", messages=MESSAGES)
        assert client._openai.chat.completions.create.call_args.kwargs["messages"] == MESSAGES

        server.replies["ai.request"] = [{"status": 200, "json": {
            "decision": "redact", "violations": [], "redactions": [{"type": "contains_pii", "count": 1}],
        }}]
        decision = client._send_request("r", {"model": "m", "messages": MESSAGES}, {})
        assert decision.redactions == [{"type": "contains_pii", "count": 1}]


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def anthropic_client(server: Server):
    from signalvault import AnthropicSignalVaultClient

    mock_anthropic = MagicMock()
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        client = AnthropicSignalVaultClient(api_key="k", anthropic_api_key="sk-ant")
    client._http = httpx.Client(transport=httpx.MockTransport(server.sync_handler))
    response = MagicMock()
    response.content = [
        MagicMock(type="text", text="Let me check. "),
        MagicMock(type="tool_use", text=None),
        MagicMock(type="text", text="Done."),
    ]
    response.usage = MagicMock(input_tokens=5, output_tokens=2)
    client._anthropic.messages.create.return_value = response
    return client


class TestAnthropic:
    def test_system_prompt_is_scanned_and_provider_call_unchanged(self):
        server = Server({"ai.request": [ALLOW]})
        client = anthropic_client(server)
        client.messages.create(model="claude", max_tokens=5, system="SYS sk-secret", messages=MESSAGES)
        client.flush()

        assert server.bodies[0]["payload"]["messages"][0] == {"role": "system", "content": "SYS sk-secret"}
        sent = client._anthropic.messages.create.call_args.kwargs
        assert sent["system"] == "SYS sk-secret"
        assert sent["messages"] == MESSAGES

    def test_records_every_text_block(self):
        server = Server({"ai.request": [ALLOW]})
        client = anthropic_client(server)
        client.messages.create(model="claude", max_tokens=5, messages=MESSAGES)
        client.flush()
        assert server.bodies[1]["payload"]["output"] == "Let me check. Done."


# ---------------------------------------------------------------------------
# Background delivery
# ---------------------------------------------------------------------------

class TestBackgroundDelivery:
    def test_retries_once_on_503_with_same_event_id(self):
        server = Server({"ai.request": [ALLOW], "ai.response": [{"status": 503}, {"status": 200}]})
        client = sync_client(server)
        client.chat.completions.create(model="m", messages=MESSAGES)
        client.flush()

        responses = [b for b in server.bodies if b["type"] == "ai.response"]
        assert len(responses) == 2
        assert responses[0]["event_id"] == responses[1]["event_id"]

    def test_short_retry_after_retried_long_one_dropped_with_warning(self):
        server = Server({
            "ai.request": [ALLOW, ALLOW],
            "ai.response": [
                {"status": 429, "headers": {"retry-after": "0"}},
                {"status": 200},
                {"status": 429, "headers": {"retry-after": "60"}},
            ],
        })
        client = sync_client(server)
        client.chat.completions.create(model="m", messages=MESSAGES)
        client.flush()
        assert server.types().count("ai.response") == 2

        with pytest.warns(SignalVaultWarning, match="being dropped"):
            client.chat.completions.create(model="m", messages=MESSAGES)
            client.flush()
        assert server.types().count("ai.response") == 3

    def test_flush_waits_for_pending_events(self):
        server = Server(delay=0.1)
        client = sync_client(server)
        client._fire_response("r", "m", "out", 1, 1, {})
        assert server.requests == []
        client.flush()
        assert server.types() == ["ai.response"]

    def test_tool_input_captured_at_call_time(self):
        """Mutating the argument after the call must not change the recorded input."""
        server = Server()
        client = sync_client(server)
        held = []
        # Hold the background job until after the mutation, so the test is deterministic.
        with patch.object(client, "_submit", side_effect=lambda fn, *a: held.append((fn, a))):
            args = {"city": "London"}
            client.tool("lookup", lambda a: "ok")(args)
            args["city"] = "MUTATED"
        fn, a = held[0]
        fn(*a)
        assert server.bodies[0]["payload"]["tool_input"] == {"city": "London"}

    def test_invalid_tool_name_does_not_break_the_tool(self):
        server = Server()
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning, match="tool_call event dropped"):
            assert client.tool("", lambda: 42)() == 42


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------

class TestAsync:
    @pytest.mark.asyncio
    async def test_background_tasks_are_referenced_and_flushed(self):
        server = Server({"ai.request": [ALLOW]}, delay=0.05)
        client = AsyncSignalVaultClient(api_key="k", openai_api_key="x")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(server.async_handler))
        client._openai = MagicMock()

        async def create(**_):
            return fake_completion()

        client._openai.chat.completions.create = create
        await client.chat.completions.create(model="m", messages=MESSAGES)
        assert len(client._tasks) == 1
        await client.flush()
        assert client._tasks == set()
        assert server.types() == ["ai.request", "ai.response"]

    def test_fire_without_loop_warns_instead_of_silently_dropping(self):
        client = AsyncSignalVaultClient(api_key="k", openai_api_key="x")
        with pytest.warns(SignalVaultWarning, match="No running event loop"):
            client._fire_tool_call(signalvault.ToolRecordOptions(tool_name="t"))


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

def test_deep_nesting_does_not_raise():
    value = cur = {}
    for _ in range(5000):
        cur["a"] = {}
        cur = cur["a"]
    result = sanitize_payload(value)
    assert "[MaxDepth]" in json.dumps(result)
