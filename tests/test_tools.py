"""Unit tests for agent tool-use capture (sync + async)."""

from __future__ import annotations

import asyncio
import json
import warnings
from typing import Any
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from signalvault import (
    AsyncSignalVaultClient,
    SignalVaultClient,
)
from signalvault.tools import (
    MAX_PAYLOAD_BYTES,
    sanitize_error,
    sanitize_payload,
    validate_tool_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int = 200, body: dict | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body or {}
    return resp


def _post_json_calls(http_mock: MagicMock) -> list[dict]:
    """Returns the list of JSON bodies POSTed via http.post."""
    return [call.kwargs["json"] for call in http_mock.post.call_args_list]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestValidateToolName:
    def test_accepts_normal_name(self):
        assert validate_tool_name("fetch_weather") == "fetch_weather"

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_tool_name("")

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            validate_tool_name(None)
        with pytest.raises(ValueError):
            validate_tool_name(123)

    def test_truncates_oversized(self):
        long = "n" * 500
        result = validate_tool_name(long)
        assert len(result.encode("utf-8")) <= 200
        assert "truncated" in result


class TestSanitizeError:
    def test_none_passes_through(self):
        assert sanitize_error(None) is None

    def test_truncates_long(self):
        long = "x" * 5000
        out = sanitize_error(long)
        assert out is not None
        assert len(out.encode("utf-8")) <= 1900
        assert "truncated" in out

    def test_stringifies_exception(self):
        out = sanitize_error(RuntimeError("boom"))
        assert out == "boom"


class TestSanitizePayload:
    def test_none(self):
        assert sanitize_payload(None) is None

    def test_primitives(self):
        assert sanitize_payload(42) == 42
        assert sanitize_payload("hello") == "hello"
        assert sanitize_payload(True) is True

    def test_dict_and_list(self):
        assert sanitize_payload({"a": [1, 2]}) == {"a": [1, 2]}

    def test_circular_reference(self):
        d: dict[str, Any] = {"name": "root"}
        d["self"] = d
        out = sanitize_payload(d)
        assert out == {"name": "root", "self": "[Circular]"}

    def test_nan_and_infinity_stringified(self):
        out = sanitize_payload({"x": float("nan"), "y": float("inf")})
        assert out == {"x": "NaN", "y": "Infinity"}

    def test_bytes_decoded_when_utf8(self):
        assert sanitize_payload(b"hello") == "hello"

    def test_oversized_returns_truncation_envelope(self):
        big = {"blob": "x" * (300 * 1024)}  # ~300 KB > 256 KB
        out = sanitize_payload(big)
        assert isinstance(out, dict)
        assert out["_signalvault_truncated"] is True
        assert out["_signalvault_original_bytes"] > MAX_PAYLOAD_BYTES
        assert isinstance(out["preview"], str)

    def test_unrepresentable_falls_back_to_repr(self):
        class Weird:
            def __repr__(self):
                return "<weird>"

        out = sanitize_payload({"obj": Weird()})
        assert out == {"obj": "<weird>"}


# ---------------------------------------------------------------------------
# Sync wrapper + manual API
# ---------------------------------------------------------------------------

@pytest.fixture
def sync_client(monkeypatch):
    client = SignalVaultClient(
        api_key="sk_test_abc",
        openai_api_key="sk-fake",
        base_url="https://api.example.com",
    )
    client._http = MagicMock()
    client._http.post.return_value = _make_response(200)
    # Make tool_call submissions synchronous so tests don't race the executor.
    # The wrapper still hits ``_fire_tool_call`` so we exercise the same path,
    # we just bypass the thread hop.
    monkeypatch.setattr(client, "_fire_tool_call", client._send_tool_call_event)
    return client


class TestSyncWrapper:
    def test_records_successful_call(self, sync_client):
        @sync_client.tool("fetch_weather")
        def fetch_weather(city: str):
            return {"temp": 12.3, "for": city}

        result = fetch_weather("London")
        assert result == {"temp": 12.3, "for": "London"}

        bodies = _post_json_calls(sync_client._http)
        assert len(bodies) == 1
        body = bodies[0]
        assert body["type"] == "agent.tool_call"
        assert body["payload"]["tool_name"] == "fetch_weather"
        assert body["payload"]["tool_input"] == "London"
        assert body["payload"]["tool_output"] == {"temp": 12.3, "for": "London"}
        assert body["payload"]["duration_ms"] >= 0
        assert "started_at" in body["payload"]

    def test_records_error_and_reraises(self, sync_client):
        @sync_client.tool("fail")
        def failing():
            raise RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            failing()

        body = _post_json_calls(sync_client._http)[0]
        assert body["payload"]["error"] == "connection refused"
        assert body["payload"]["tool_output"] is None

    def test_single_arg_stored_as_value_not_array(self, sync_client):
        @sync_client.tool("search")
        def search(query: dict):
            return {"hits": query["k"]}

        search({"q": "embeddings", "k": 5})
        body = _post_json_calls(sync_client._http)[0]
        assert body["payload"]["tool_input"] == {"q": "embeddings", "k": 5}

    def test_multi_arg_stored_as_args_kwargs(self, sync_client):
        @sync_client.tool("add")
        def add(a: int, b: int) -> int:
            return a + b

        add(2, 3)
        body = _post_json_calls(sync_client._http)[0]
        assert body["payload"]["tool_input"] == {"args": [2, 3], "kwargs": {}}

    def test_kwargs_only_stored_as_dict(self, sync_client):
        @sync_client.tool("named")
        def named(**kwargs: Any) -> Any:
            return kwargs

        named(x=1, y=2)
        body = _post_json_calls(sync_client._http)[0]
        assert body["payload"]["tool_input"] == {"x": 1, "y": 2}

    def test_decorator_form(self, sync_client):
        @sync_client.tool("decorated")
        def fn(v: int) -> int:
            return v * 2

        assert fn(5) == 10
        body = _post_json_calls(sync_client._http)[0]
        assert body["payload"]["tool_name"] == "decorated"


class TestSyncToolsRecord:
    def test_posts_with_explicit_request_id(self, sync_client):
        sync_client.tools.record(
            tool_name="manual_tool",
            tool_input={"x": 1},
            tool_output={"y": 2},
            duration_ms=50,
            request_id="req-explicit",
        )
        body = _post_json_calls(sync_client._http)[0]
        assert body["type"] == "agent.tool_call"
        assert body["request_id"] == "req-explicit"
        assert body["payload"]["tool_name"] == "manual_tool"
        assert body["payload"]["duration_ms"] == 50

    def test_omits_request_id_when_none(self, sync_client):
        sync_client.tools.record(tool_name="orphan", duration_ms=10)
        body = _post_json_calls(sync_client._http)[0]
        assert "request_id" not in body

    def test_rejects_empty_tool_name(self, sync_client):
        with pytest.raises(ValueError, match="non-empty string"):
            sync_client.tools.record(tool_name="", duration_ms=1)
        assert sync_client._http.post.call_count == 0

    def test_truncates_long_tool_name(self, sync_client):
        sync_client.tools.record(tool_name="x" * 500, duration_ms=1)
        body = _post_json_calls(sync_client._http)[0]
        assert len(body["payload"]["tool_name"].encode("utf-8")) <= 200
        assert "truncated" in body["payload"]["tool_name"]


class TestSyncWithContext:
    def test_auto_links_wrapped_tool_calls(self, sync_client):
        @sync_client.tool("search")
        def search(q: str):
            return {"q": q}

        with sync_client.with_context(request_id="turn-abc"):
            search("embeddings")

        body = _post_json_calls(sync_client._http)[0]
        assert body["request_id"] == "turn-abc"

    def test_explicit_overrides_context(self, sync_client):
        with sync_client.with_context(request_id="ctx-id"):
            sync_client.tools.record(
                tool_name="explicit",
                duration_ms=1,
                request_id="explicit-id",
            )
        body = _post_json_calls(sync_client._http)[0]
        assert body["request_id"] == "explicit-id"

    def test_outside_any_context_is_orphan(self, sync_client):
        @sync_client.tool("orphan_tool")
        def fn():
            return "ok"

        fn()
        body = _post_json_calls(sync_client._http)[0]
        assert "request_id" not in body


class TestSyncSanitization:
    def test_circular_reference_not_crash(self, sync_client):
        circular: dict[str, Any] = {"name": "root"}
        circular["self"] = circular

        @sync_client.tool("circular_tool")
        def fn(_arg: Any):
            return "ok"

        fn(circular)
        body = _post_json_calls(sync_client._http)[0]
        assert body["payload"]["tool_input"] == {"name": "root", "self": "[Circular]"}

    def test_oversized_output_truncated(self, sync_client):
        big = "x" * (300 * 1024)

        @sync_client.tool("big_output")
        def fn():
            return {"blob": big}

        fn()
        body = _post_json_calls(sync_client._http)[0]
        out = body["payload"]["tool_output"]
        assert out["_signalvault_truncated"] is True
        assert out["_signalvault_original_bytes"] > MAX_PAYLOAD_BYTES

    def test_long_error_truncated(self, sync_client):
        long_msg = "a" * 5000

        @sync_client.tool("long_error")
        def fn():
            raise RuntimeError(long_msg)

        with pytest.raises(RuntimeError):
            fn()

        body = _post_json_calls(sync_client._http)[0]
        err = body["payload"]["error"]
        assert len(err.encode("utf-8")) <= 1900
        assert "truncated" in err


class TestSyncErrorLogging:
    def test_4xx_warns_unconditionally(self, sync_client):
        sync_client._http.post.return_value = _make_response(422)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sync_client.tools.record(tool_name="will_fail", duration_ms=1)
        msgs = [str(w.message) for w in caught]
        assert any("422" in m for m in msgs), f"expected 422 warning, got {msgs}"
        assert any("api_key" in m for m in msgs)

    def test_5xx_silent_unless_debug(self, sync_client):
        sync_client._http.post.return_value = _make_response(503)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sync_client.tools.record(tool_name="svc_down", duration_ms=1)
        msgs = [str(w.message) for w in caught]
        assert not any("503" in m for m in msgs)


# ---------------------------------------------------------------------------
# Async wrapper + manual API
# ---------------------------------------------------------------------------

@pytest.fixture
def async_client():
    client = AsyncSignalVaultClient(
        api_key="sk_test_abc",
        openai_api_key="sk-fake",
        base_url="https://api.example.com",
    )
    client._http = MagicMock()
    client._http.post = AsyncMock(return_value=_make_response(200))
    return client


class TestAsyncWrapper:
    async def test_records_successful_call(self, async_client):
        @async_client.tool("fetch_weather")
        async def fetch_weather(city: str):
            return {"temp": 12.3, "for": city}

        result = await fetch_weather("London")
        assert result == {"temp": 12.3, "for": "London"}

        # Let the create_task'd audit fire.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        bodies = _post_json_calls(async_client._http)
        assert len(bodies) == 1
        body = bodies[0]
        assert body["type"] == "agent.tool_call"
        assert body["payload"]["tool_input"] == "London"
        assert body["payload"]["tool_output"] == {"temp": 12.3, "for": "London"}

    async def test_records_error_and_reraises(self, async_client):
        @async_client.tool("fail")
        async def failing():
            raise RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            await failing()

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        body = _post_json_calls(async_client._http)[0]
        assert body["payload"]["error"] == "connection refused"


class TestAsyncToolsRecord:
    async def test_posts_event(self, async_client):
        await async_client.tools.record(
            tool_name="manual_tool",
            tool_input={"x": 1},
            duration_ms=50,
            request_id="req-explicit",
        )
        body = _post_json_calls(async_client._http)[0]
        assert body["request_id"] == "req-explicit"
        assert body["payload"]["tool_name"] == "manual_tool"


class TestAsyncWithContext:
    async def test_auto_links_wrapped_tool_calls(self, async_client):
        @async_client.tool("search")
        async def search(q: str):
            return {"q": q}

        async with async_client.with_context(request_id="turn-abc"):
            await search("embeddings")

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        body = _post_json_calls(async_client._http)[0]
        assert body["request_id"] == "turn-abc"

    async def test_explicit_overrides_context(self, async_client):
        async with async_client.with_context(request_id="ctx-id"):
            await async_client.tools.record(
                tool_name="explicit",
                duration_ms=1,
                request_id="explicit-id",
            )
        body = _post_json_calls(async_client._http)[0]
        assert body["request_id"] == "explicit-id"
