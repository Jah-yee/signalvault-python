"""Unit tests for SignalVault Python SDK (no network calls)."""

import pytest
import warnings
from unittest.mock import MagicMock, patch, AsyncMock
from signalvault import (
    SignalVaultWarning,
    SignalVaultClient,
    AsyncSignalVaultClient,
    AnthropicSignalVaultClient,
    AsyncAnthropicSignalVaultClient,
)
from signalvault.client import Decision, _merge_metadata, _parse_decision, Violation


# ---------------------------------------------------------------------------
# SignalVaultClient (sync, OpenAI)
# ---------------------------------------------------------------------------

class TestSignalVaultClientConstruction:
    def test_basic_construction(self):
        client = SignalVaultClient(api_key="sk_test_abc", openai_api_key="sk-fake")
        assert client is not None
        assert client.chat is not None
        assert client.chat.completions is not None

    def test_defaults(self):
        client = SignalVaultClient(api_key="sk_test_abc", openai_api_key="sk-fake")
        assert client._config.base_url == "https://api.signalvault.io"
        assert client._config.fail_mode == "open"
        assert client._config.environment == "production"
        assert client._config.debug is False
        assert client._config.mirror_mode is False
        assert client._config.preflight_timeout == 2.0
        assert client._config.timeout == 30.0
        assert client._config.metadata == {}

    def test_custom_config(self):
        client = SignalVaultClient(
            api_key="sk_test_abc",
            openai_api_key="sk-fake",
            base_url="https://api.signalvault.io/",
            environment="staging",
            debug=True,
            mirror_mode=True,
            preflight_timeout=0.5,
            timeout=5.0,
            metadata={"user_id": "u_123"},
        )
        assert client._config.base_url == "https://api.signalvault.io"
        assert client._config.environment == "staging"
        assert client._config.debug is True
        assert client._config.mirror_mode is True
        assert client._config.preflight_timeout == 0.5
        assert client._config.timeout == 5.0
        assert client._config.metadata == {"user_id": "u_123"}

    def test_strips_trailing_slash(self):
        client = SignalVaultClient(
            api_key="sk_test_abc", openai_api_key="sk-fake",
            base_url="https://api.signalvault.io///",
        )
        assert client._config.base_url == "https://api.signalvault.io"

    def test_strips_no_trailing_slash(self):
        client = SignalVaultClient(
            api_key="sk_test_abc", openai_api_key="sk-fake",
            base_url="https://api.signalvault.io",
        )
        assert client._config.base_url == "https://api.signalvault.io"


class TestMetadata:
    def test_merge_metadata_config_only(self):
        result = _merge_metadata({"user_id": "u_1"}, None)
        assert result == {"user_id": "u_1"}

    def test_merge_metadata_call_overrides(self):
        result = _merge_metadata({"user_id": "u_1", "env": "prod"}, {"user_id": "u_2", "feature": "chat"})
        assert result == {"user_id": "u_2", "env": "prod", "feature": "chat"}

    def test_merge_metadata_empty(self):
        result = _merge_metadata({}, {})
        assert result == {}


class TestParseDecision:
    def test_basic(self):
        d = _parse_decision({"decision": "block", "violations": [], "redactions": []})
        assert d.decision == "block"
        assert d.violations == []

    def test_unknown_violation_fields_do_not_crash(self):
        """Server may add new fields to violations in future — must not crash."""
        data = {
            "decision": "warn",
            "violations": [
                {
                    "rule_id": "r1",
                    "type": "pii",
                    "severity": 2,
                    "action": "warn",
                    "details": {},
                    "future_field": "some_new_value",  # unknown field
                }
            ],
            "redactions": [],
        }
        d = _parse_decision(data)
        assert len(d.violations) == 1
        assert d.violations[0].rule_id == "r1"
        assert not hasattr(d.violations[0], "future_field")

    def test_defaults_on_empty(self):
        d = _parse_decision({})
        assert d.decision == "allow"
        assert d.violations == []
        assert d.redactions == []


class TestTimeout:
    def test_preflight_timeout_default(self):
        client = SignalVaultClient(api_key="sk_test", openai_api_key="sk-fake")
        assert client._config.preflight_timeout == 2.0

    def test_custom_preflight_timeout(self):
        client = SignalVaultClient(
            api_key="sk_test", openai_api_key="sk-fake", preflight_timeout=0.5
        )
        assert client._config.preflight_timeout == 0.5

    def test_fail_open_on_connection_error(self):
        client = SignalVaultClient(
            api_key="sk_test", openai_api_key="sk-fake",
            base_url="http://127.0.0.1:1",  # unreachable
            preflight_timeout=0.1,
        )
        with pytest.warns(SignalVaultWarning, match="NOT applied"):
            decision = client._send_request("req-1", {"model": "gpt-4", "messages": []}, {})
        assert decision.decision == "allow"
        assert decision.violations == []


class TestStreamingNotImplemented:
    def test_streaming_no_longer_raises(self):
        """Streaming was previously NotImplementedError — now it should be implemented."""
        client = SignalVaultClient(api_key="sk_test_abc", openai_api_key="sk-fake")
        import inspect
        completions = client.chat.completions
        assert hasattr(completions, "create")
        source = inspect.getsource(completions.create)
        assert "NotImplementedError" not in source


# ---------------------------------------------------------------------------
# sv_metadata
# ---------------------------------------------------------------------------

class TestSvMetadata:
    def test_sv_metadata_strips_and_merges(self):
        """sv_metadata is removed from OpenAI kwargs and merged with config metadata."""
        client = SignalVaultClient(
            api_key="sk_test", openai_api_key="sk-fake",
            metadata={"env": "prod"},
        )
        captured_oai_kwargs = {}
        captured_request_metadata = {}

        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "hello"
        fake_response.usage = MagicMock(prompt_tokens=5, completion_tokens=10)

        def fake_oai_create(**kwargs):
            captured_oai_kwargs.update(kwargs)
            return fake_response

        def fake_send_request(request_id, params, metadata):
            captured_request_metadata.update(metadata)
            return Decision()

        with patch.object(client._openai.chat.completions, "create", side_effect=fake_oai_create), \
             patch.object(client, "_send_request", side_effect=fake_send_request), \
             patch.object(client, "_fire_response"):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                sv_metadata={"tool": "clip_detect"},
            )

        assert "sv_metadata" not in captured_oai_kwargs
        assert captured_request_metadata == {"env": "prod", "tool": "clip_detect"}

    def test_provider_metadata_passes_through(self):
        """`metadata` belongs to OpenAI (stored completions) and is forwarded untouched."""
        client = SignalVaultClient(api_key="sk_test", openai_api_key="sk-fake")
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "hello"
        fake_response.usage = MagicMock(prompt_tokens=5, completion_tokens=10)
        captured_request_metadata = {}

        def fake_send_request(request_id, params, metadata):
            captured_request_metadata.update(metadata)
            return Decision()

        with patch.object(client._openai.chat.completions, "create", return_value=fake_response) as oai, \
             patch.object(client, "_send_request", side_effect=fake_send_request), \
             patch.object(client, "_fire_response"), \
             warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                metadata={"purpose": "eval"},
                sv_metadata={"tool": "x"},
            )

        assert oai.call_args.kwargs["metadata"] == {"purpose": "eval"}
        assert captured_request_metadata == {"tool": "x"}


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class TestStreaming:
    def _make_chunk(self, content=None, prompt_tokens=None, completion_tokens=None):
        chunk = MagicMock()
        if content is not None:
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = content
        else:
            chunk.choices = []
        if prompt_tokens is not None:
            chunk.usage = MagicMock(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
        else:
            chunk.usage = None
        return chunk

    def test_streaming_sends_response_event(self):
        """After consuming all stream chunks, _fire_response is called with correct data."""
        client = SignalVaultClient(api_key="sk_test", openai_api_key="sk-fake")

        chunks = [
            self._make_chunk("Hello"),
            self._make_chunk(" world"),
            self._make_chunk(prompt_tokens=10, completion_tokens=5),
        ]

        with patch.object(client, "_send_request", return_value=Decision()), \
             patch.object(client._openai.chat.completions, "create", return_value=iter(chunks)), \
             patch.object(client, "_fire_response") as mock_fire:
            gen = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            list(gen)  # consume all

        mock_fire.assert_called_once()
        args = mock_fire.call_args[0]
        # (request_id, model, output, prompt_tokens, completion_tokens, metadata)
        assert args[2] == "Hello world"
        assert args[3] == 10
        assert args[4] == 5

    def test_streaming_finally_sends_on_break(self):
        """Breaking out of stream loop early still fires the response event."""
        client = SignalVaultClient(api_key="sk_test", openai_api_key="sk-fake")

        chunks = [
            self._make_chunk("Hello"),
            self._make_chunk(" more"),
        ]

        with patch.object(client, "_send_request", return_value=Decision()), \
             patch.object(client._openai.chat.completions, "create", return_value=iter(chunks)), \
             patch.object(client, "_fire_response") as mock_fire:
            gen = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            for _ in gen:
                break  # early exit
            gen.close()  # explicitly close to trigger finally block

        mock_fire.assert_called_once()
        args = mock_fire.call_args[0]
        assert args[2] == "Hello"  # only the first chunk was accumulated


# ---------------------------------------------------------------------------
# mirror_mode
# ---------------------------------------------------------------------------

class TestMirrorMode:
    def test_mirror_mode_sends_request_before_response(self):
        """_send_audit_from_parts sends ai.request sequentially before ai.response."""
        client = SignalVaultClient(api_key="sk_test", openai_api_key="sk-fake")

        call_order = []

        def mock_post(url, **kwargs):
            event_type = kwargs.get("json", {}).get("type")
            call_order.append(event_type)
            resp = MagicMock()
            resp.status_code = 200
            return resp

        with patch.object(client._http, "post", side_effect=mock_post):
            client._send_audit_from_parts(
                "req-1", "gpt-4", [{"role": "user", "content": "hi"}],
                "hello output", 5, 10, {}
            )

        assert call_order == ["ai.request", "ai.response"]

    def test_mirror_mode_nonblocking(self):
        """_mirror() submits audit via _fire_audit and returns the response immediately."""
        client = SignalVaultClient(
            api_key="sk_test", openai_api_key="sk-fake", mirror_mode=True
        )
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "hello"
        fake_response.usage = MagicMock(prompt_tokens=5, completion_tokens=10)

        with patch.object(client._openai.chat.completions, "create", return_value=fake_response), \
             patch.object(client, "_fire_audit") as mock_fire:
            result = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )

        mock_fire.assert_called_once()
        assert result is fake_response


# ---------------------------------------------------------------------------
# warn decision
# ---------------------------------------------------------------------------

class TestWarnDecision:
    def test_sync_warn_decision_emits_warning_in_debug(self):
        """decision='warn' with debug=True emits a UserWarning."""
        client = SignalVaultClient(api_key="sk_test", openai_api_key="sk-fake", debug=True)
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "hello"
        fake_response.usage = MagicMock(prompt_tokens=5, completion_tokens=10)

        warn_decision = Decision(decision="warn", violations=[])

        with patch.object(client, "_send_request", return_value=warn_decision), \
             patch.object(client._openai.chat.completions, "create", return_value=fake_response), \
             patch.object(client, "_fire_response"), \
             pytest.warns(UserWarning, match="Warnings"):
            client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_sync_warn_decision_no_warning_without_debug(self):
        """decision='warn' without debug=True does not emit a warning."""
        client = SignalVaultClient(api_key="sk_test", openai_api_key="sk-fake", debug=False)
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "hello"
        fake_response.usage = MagicMock(prompt_tokens=5, completion_tokens=10)

        warn_decision = Decision(decision="warn", violations=[])

        with patch.object(client, "_send_request", return_value=warn_decision), \
             patch.object(client._openai.chat.completions, "create", return_value=fake_response), \
             patch.object(client, "_fire_response"):
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                # Should not raise
                client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "hi"}],
                )

    @pytest.mark.asyncio
    async def test_async_warn_decision_emits_warning_in_debug(self):
        """Async client: decision='warn' with debug=True emits a UserWarning."""
        client = AsyncSignalVaultClient(api_key="sk_test", openai_api_key="sk-fake", debug=True)
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "hello"
        fake_response.usage = MagicMock(prompt_tokens=5, completion_tokens=10)

        warn_decision = Decision(decision="warn", violations=[])

        with patch.object(client, "_send_request", new=AsyncMock(return_value=warn_decision)), \
             patch.object(client._openai.chat.completions, "create", new=AsyncMock(return_value=fake_response)), \
             patch.object(client, "_send_response_from_parts", new=AsyncMock()), \
             pytest.warns(UserWarning, match="Warnings"):
            await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
            )


# ---------------------------------------------------------------------------
# Resource management — close() / context managers
# ---------------------------------------------------------------------------

class TestResourceManagement:
    def test_sync_client_has_close(self):
        client = SignalVaultClient(api_key="sk_test", openai_api_key="sk-fake")
        assert callable(getattr(client, "close", None))

    def test_sync_context_manager(self):
        with SignalVaultClient(api_key="sk_test", openai_api_key="sk-fake") as client:
            assert client is not None
            assert client.chat is not None

    def test_async_client_has_aclose(self):
        client = AsyncSignalVaultClient(api_key="sk_test", openai_api_key="sk-fake")
        assert callable(getattr(client, "aclose", None))

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        async with AsyncSignalVaultClient(api_key="sk_test", openai_api_key="sk-fake") as client:
            assert client is not None
            assert client.chat is not None

    def test_anthropic_sync_context_manager(self):
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            with AnthropicSignalVaultClient(
                api_key="sk_test", anthropic_api_key="sk-ant-fake"
            ) as client:
                assert client is not None

    @pytest.mark.asyncio
    async def test_anthropic_async_context_manager(self):
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            async with AsyncAnthropicSignalVaultClient(
                api_key="sk_test", anthropic_api_key="sk-ant-fake"
            ) as client:
                assert client is not None


# ---------------------------------------------------------------------------
# AsyncSignalVaultClient
# ---------------------------------------------------------------------------

class TestAsyncSignalVaultClient:
    def test_construction(self):
        client = AsyncSignalVaultClient(api_key="sk_test", openai_api_key="sk-fake")
        assert client is not None
        assert client.chat is not None
        assert client.chat.completions is not None

    def test_defaults(self):
        client = AsyncSignalVaultClient(api_key="sk_test", openai_api_key="sk-fake")
        assert client._config.preflight_timeout == 2.0
        assert client._config.timeout == 30.0

    def test_strips_trailing_slash(self):
        client = AsyncSignalVaultClient(
            api_key="sk_test", openai_api_key="sk-fake",
            base_url="https://api.signalvault.io///",
        )
        assert client._config.base_url == "https://api.signalvault.io"

    @pytest.mark.asyncio
    async def test_fail_open_on_timeout(self):
        client = AsyncSignalVaultClient(
            api_key="sk_test", openai_api_key="sk-fake",
            base_url="http://127.0.0.1:1",
            preflight_timeout=0.1,
        )
        with pytest.warns(SignalVaultWarning, match="NOT applied"):
            decision = await client._send_request("req-1", {"model": "gpt-4", "messages": []}, {})
        assert decision.decision == "allow"


# ---------------------------------------------------------------------------
# AnthropicSignalVaultClient
# ---------------------------------------------------------------------------

class TestAnthropicSignalVaultClient:
    def test_raises_on_missing_sdk(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="pip install signalvault"):
                AnthropicSignalVaultClient(
                    api_key="sk_test", anthropic_api_key="sk-ant-fake"
                )

    def test_construction_with_mock_sdk(self):
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicSignalVaultClient(
                api_key="sk_test", anthropic_api_key="sk-ant-fake"
            )
            assert client is not None
            assert client.messages is not None
            assert client._config.preflight_timeout == 2.0

    def test_metadata_config(self):
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicSignalVaultClient(
                api_key="sk_test",
                anthropic_api_key="sk-ant-fake",
                metadata={"workspace": "ws_1"},
            )
            assert client._config.metadata == {"workspace": "ws_1"}

    def test_strips_trailing_slash(self):
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicSignalVaultClient(
                api_key="sk_test", anthropic_api_key="sk-ant-fake",
                base_url="https://api.signalvault.io///",
            )
            assert client._config.base_url == "https://api.signalvault.io"

    def test_fail_open_on_connection_error(self):
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicSignalVaultClient(
                api_key="sk_test", anthropic_api_key="sk-ant-fake",
                base_url="http://127.0.0.1:1",
                preflight_timeout=0.1,
            )
            with pytest.warns(SignalVaultWarning, match="NOT applied"):
                decision = client._send_request("req-1", {"model": "claude-3-5-sonnet-20241022", "messages": []}, {})
            assert decision.decision == "allow"

    def test_provider_metadata_passes_through(self):
        """Anthropic's `metadata={"user_id": ...}` reaches Anthropic instead of being taken over."""
        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AnthropicSignalVaultClient(api_key="sk_test", anthropic_api_key="sk-ant-fake")
            fake_response = MagicMock()
            fake_response.content = [MagicMock(type="text", text="hi")]
            fake_response.usage = MagicMock(input_tokens=5, output_tokens=10)

            with patch.object(client._anthropic.messages, "create", return_value=fake_response) as create, \
                 patch.object(client, "_send_request", return_value=Decision()), \
                 patch.object(client, "_fire_response"):
                client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=100,
                    metadata={"user_id": "end-user-42"},
                )

            assert create.call_args.kwargs["metadata"] == {"user_id": "end-user-42"}


# ---------------------------------------------------------------------------
# AsyncAnthropicSignalVaultClient
# ---------------------------------------------------------------------------

class TestAsyncAnthropicSignalVaultClient:
    def test_raises_on_missing_sdk(self):
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="pip install signalvault"):
                AsyncAnthropicSignalVaultClient(
                    api_key="sk_test", anthropic_api_key="sk-ant-fake"
                )

    def test_construction_with_mock_sdk(self):
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AsyncAnthropicSignalVaultClient(
                api_key="sk_test", anthropic_api_key="sk-ant-fake"
            )
            assert client is not None
            assert client.messages is not None

    def test_strips_trailing_slash(self):
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AsyncAnthropicSignalVaultClient(
                api_key="sk_test", anthropic_api_key="sk-ant-fake",
                base_url="https://api.signalvault.io///",
            )
            assert client._config.base_url == "https://api.signalvault.io"

    @pytest.mark.asyncio
    async def test_fail_open_on_timeout(self):
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            client = AsyncAnthropicSignalVaultClient(
                api_key="sk_test", anthropic_api_key="sk-ant-fake",
                base_url="http://127.0.0.1:1",
                preflight_timeout=0.1,
            )
            with pytest.warns(SignalVaultWarning, match="NOT applied"):
                decision = await client._send_request(
                    "req-1", {"model": "claude-3-5-sonnet-20241022", "messages": []}, {}
                )
            assert decision.decision == "allow"
