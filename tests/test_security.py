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


# ---------------------------------------------------------------------------
# Review follow-ups
# ---------------------------------------------------------------------------

class TestMalformedDecisions:
    @pytest.mark.parametrize("body", [
        {"decision": ["block"]},
        {"decision": {}},
        {"decision": "allow", "violations": 5},
    ])
    def test_wrong_types_fail_open_instead_of_raising(self, body):
        server = Server({"ai.request": [{"status": 200, "json": body}]})
        client = sync_client(server)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            client.chat.completions.create(model="m", messages=MESSAGES)
        client._openai.chat.completions.create.assert_called_once()

    def test_non_string_violation_type_in_block(self):
        server = Server({"ai.request": [{"status": 200, "json": {
            "decision": "block", "violations": [{"type": 7}, "junk", {"type": "pii"}],
        }}]})
        client = sync_client(server)
        with pytest.raises(SignalVaultBlockedError, match=r"\(7, pii\)"):
            client.chat.completions.create(model="m", messages=MESSAGES)


class TestWarningsRepeat:
    def test_repeats_after_the_window_under_default_filters(self, monkeypatch):
        import signalvault.client as client_module

        monkeypatch.setattr(client_module, "_WARN_INTERVAL_SECONDS", 0.0)
        server = Server({"ai.request": [{"status": 401, "json": {}} for _ in range(3)]})
        client = sync_client(server)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            for _ in range(3):
                client._send_request("r", {"model": "m", "messages": []}, {})
        assert len(caught) == 3

    def test_points_at_the_callers_code(self):
        server = Server({"ai.request": [{"status": 401, "json": {}}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning) as record:
            client.chat.completions.create(model="m", messages=MESSAGES)
        assert record[0].filename == __file__


class TestRetryPolicy:
    def test_warns_when_the_retry_is_rate_limited_too(self):
        server = Server({
            "ai.request": [ALLOW],
            "ai.response": [
                {"status": 429, "headers": {"retry-after": "0"}},
                {"status": 429, "headers": {"retry-after": "60"}},
            ],
        })
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning, match="being dropped"):
            client.chat.completions.create(model="m", messages=MESSAGES)
            client.flush()
        assert server.types().count("ai.response") == 2

    def test_timeouts_are_not_retried(self):
        attempts = []

        def handler(request):
            attempts.append(json.loads(request.content)["type"])
            raise httpx.ReadTimeout("slow", request=request)

        client = SignalVaultClient(api_key="k", openai_api_key="x")
        client._http = httpx.Client(transport=httpx.MockTransport(handler))
        client._fire_response("r", "m", "out", 1, 1, {})
        client.flush()
        assert attempts == ["ai.response"]

    def test_manual_tool_record_does_not_retry(self):
        server = Server({"agent.tool_call": [{"status": 503}, {"status": 200}]})
        client = sync_client(server)
        client.tools.record(tool_name="t", duration_ms=1)
        assert server.types() == ["agent.tool_call"]


class TestAuditWhenPreflightFailed:
    def test_records_the_request_before_the_response_on_503(self):
        server = Server({"ai.request": [{"status": 503}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning):
            client.chat.completions.create(model="m", messages=MESSAGES)
        client.flush()

        assert server.types() == ["ai.request", "ai.request", "ai.response"]
        fallback = server.bodies[1]
        assert fallback["request_id"] == server.bodies[0]["request_id"]
        assert "event_id" not in fallback
        assert fallback["payload"] == {"messages": MESSAGES, "preflight_unavailable": True}

    def test_no_audit_events_after_a_refused_preflight(self):
        server = Server({"ai.request": [{"status": 401, "json": {}}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning):
            client.chat.completions.create(model="m", messages=MESSAGES)
        client.flush()
        assert server.types() == ["ai.request"]

    @pytest.mark.asyncio
    async def test_async_records_the_request_on_network_error(self):
        calls = []

        async def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            if len(calls) == 1:
                raise httpx.ConnectError("down", request=request)
            return httpx.Response(200, json={})

        client = AsyncSignalVaultClient(api_key="k", openai_api_key="x")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client._openai = MagicMock()

        async def create(**_):
            return fake_completion()

        client._openai.chat.completions.create = create
        with pytest.warns(SignalVaultWarning, match="unreachable"):
            await client.chat.completions.create(model="m", messages=MESSAGES)
        await client.flush()
        assert [c["type"] for c in calls] == ["ai.request", "ai.request", "ai.response"]
        assert calls[1]["payload"]["preflight_unavailable"] is True


class TestFailureClassification:
    def test_redirect_is_reported_as_a_base_url_problem(self):
        server = Server({"ai.request": [{"status": 307, "headers": {"location": "https://elsewhere"}}]})
        client = sync_client(server)
        with pytest.warns(SignalVaultWarning, match=r"redirected \(307\).*base_url"):
            client.chat.completions.create(model="m", messages=MESSAGES)

    def test_fail_closed_on_network_error(self):
        def handler(request):
            raise httpx.ConnectError("down", request=request)

        client = sync_client(Server(), fail_mode="closed")
        client._http = httpx.Client(transport=httpx.MockTransport(handler))
        with pytest.raises(SignalVaultUnavailableError, match="unreachable"):
            client.chat.completions.create(model="m", messages=MESSAGES)
        client._openai.chat.completions.create.assert_not_called()

    def test_base_url_with_query_or_fragment_is_refused(self):
        for bad in ("https://api.signalvault.io?x=1", "https://api.signalvault.io#frag"):
            with pytest.raises(ValueError, match="query or fragment"):
                normalize_base_url(bad)
