"""SignalVault client — wraps OpenAI and Anthropic with guardrails and audit logging."""

from __future__ import annotations

import asyncio
import atexit
import functools
import inspect
import time
import traceback
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Generator, List, Optional

import httpx

from .tools import (
    ToolContext,
    ToolRecordOptions,
    build_tool_call_body,
    get_current_request_id,
)


# ---------------------------------------------------------------------------
# Config & shared types
# ---------------------------------------------------------------------------

@dataclass
class SignalVaultConfig:
    api_key: str
    base_url: str = "http://localhost:4000"
    environment: str = "production"
    debug: bool = False
    mirror_mode: bool = False
    # Timeout for pre-flight /v1/events call (critical path). Fails open on timeout.
    preflight_timeout: float = 2.0
    # Timeout for background/post-flight calls.
    timeout: float = 30.0
    # Default metadata attached to every event. Merged with per-call sv_metadata.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Violation:
    rule_id: Optional[str] = None
    type: str = ""
    severity: int = 0
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    decision: str = "allow"
    violations: List[Violation] = field(default_factory=list)
    redactions: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level persistent background executor (daemon threads)
# ---------------------------------------------------------------------------

_BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sv-audit")
atexit.register(_BACKGROUND_EXECUTOR.shutdown, wait=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_metadata(config_meta: Dict[str, Any], call_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {**config_meta, **(call_meta or {})}


def _parse_decision(data: dict) -> Decision:
    known = Violation.__dataclass_fields__
    violations = [
        Violation(**{k: v for k, v in item.items() if k in known})
        for item in data.get("violations", [])
    ]
    return Decision(
        decision=data.get("decision", "allow"),
        violations=violations,
        redactions=data.get("redactions", []),
    )


# ---------------------------------------------------------------------------
# Base sync client — shared HTTP logic for OpenAI and Anthropic sync clients
# ---------------------------------------------------------------------------

class _BaseSyncClient:
    """Shared HTTP, config, and audit logic for sync SignalVault clients."""

    _provider: str = ""  # overridden by subclasses

    def __init__(self, config: SignalVaultConfig) -> None:
        self._config = config
        self._http = httpx.Client()

    # -- Resource management -------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> "_BaseSyncClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -- Internal HTTP -------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }

    def _send_request(self, request_id: str, params: dict, metadata: dict) -> Decision:
        try:
            resp = self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.preflight_timeout,
                json={
                    "type": "ai.request",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": self._provider,
                    "model": params.get("model", ""),
                    "metadata": metadata,
                    "payload": {"messages": params.get("messages", [])},
                },
            )
            if resp.status_code == 200:
                return _parse_decision(resp.json())
        except (httpx.TimeoutException, httpx.RequestError):
            if self._config.debug:
                traceback.print_exc()
        return Decision()

    def _fire_response(
        self, request_id: str, model: str, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict,
    ) -> None:
        """Submit response event to background executor — does not block caller."""
        _BACKGROUND_EXECUTOR.submit(
            self._send_response_from_parts,
            request_id, model, output, prompt_tokens, completion_tokens, metadata,
        )

    def _send_response_from_parts(
        self, request_id: str, model: str, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict,
    ) -> None:
        try:
            self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.timeout,
                json={
                    "type": "ai.response",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": self._provider,
                    "model": model,
                    "metadata": metadata,
                    "payload": {
                        "output": output,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        },
                    },
                },
            )
        except Exception:
            if self._config.debug:
                traceback.print_exc()

    def _fire_audit(
        self, request_id: str, model: str, messages: list, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict,
    ) -> None:
        """Submit audit events to background executor — does not block caller."""
        _BACKGROUND_EXECUTOR.submit(
            self._send_audit_from_parts,
            request_id, model, messages, output, prompt_tokens, completion_tokens, metadata,
        )

    def _send_audit_from_parts(
        self, request_id: str, model: str, messages: list, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict,
    ) -> None:
        """Blocking sequential HTTP sends — runs in background thread."""
        url = f"{self._config.base_url}/v1/events"
        headers = self._headers()
        timeout = self._config.timeout
        base = {
            "request_id": request_id,
            "environment": self._config.environment,
            "provider": self._provider,
            "model": model,
            "metadata": metadata,
        }

        # Send ai.request FIRST and wait for it before sending ai.response
        try:
            self._http.post(url, headers=headers, timeout=timeout, json={
                **base,
                "type": "ai.request",
                "payload": {"messages": messages, "monitor_mode": True},
            })
        except Exception:
            if self._config.debug:
                traceback.print_exc()

        try:
            self._http.post(url, headers=headers, timeout=timeout, json={
                **base,
                "type": "ai.response",
                "payload": {
                    "output": output,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    "monitor_mode": True,
                },
            })
        except Exception:
            if self._config.debug:
                traceback.print_exc()

    # -- Agent tool-use capture ---------------------------------------------

    def _send_tool_call_event(self, opts: ToolRecordOptions) -> None:
        """Blocking POST of an agent.tool_call event. Surfaces 4xx via warnings.

        Used by the manual API (``client.tools.record``) directly and by the
        wrapper indirectly via the background executor.
        """
        body = build_tool_call_body(
            environment=self._config.environment,
            default_metadata=self._config.metadata,
            opts=opts,
        )
        try:
            resp = self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.timeout,
                json=body,
            )
            _warn_on_client_error(resp.status_code, self._config.debug)
        except (httpx.TimeoutException, httpx.RequestError):
            if self._config.debug:
                traceback.print_exc()

    def _fire_tool_call(self, opts: ToolRecordOptions) -> None:
        """Submit a tool_call event to the background executor. Non-blocking."""
        _BACKGROUND_EXECUTOR.submit(self._send_tool_call_event, opts)

    @property
    def tools(self) -> "_SyncToolsAPI":
        """Manual tool-call recording API. ``client.tools.record(...)``."""
        return _SyncToolsAPI(self)

    def with_context(self, *, request_id: str) -> ToolContext:
        """Returns a context manager scoping ``request_id`` for tool correlation.

        Usable as both ``with client.with_context(request_id=...)`` and
        ``async with client.with_context(request_id=...)``.
        """
        return ToolContext(request_id)

    def tool(
        self,
        name: str,
        fn: Optional[Callable[..., Any]] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable[..., Any]:
        """Wraps a sync or async callable so each call records an agent.tool_call.

        Two call shapes — both are equivalent::

            wrapped = client.tool("fetch_weather", fetch_weather)
            wrapped = client.tool("fetch_weather")(fetch_weather)  # decorator

        For async functions used inside an async runtime, prefer
        :class:`AsyncSignalVaultClient.tool` — sync clients submit recording to
        a background thread, which is fine but blocks the event loop briefly
        when serializing arguments.
        """

        def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.iscoroutinefunction(inner):
                # Async fn through a sync client — return an async wrapper.
                @functools.wraps(inner)
                async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                    started_at, start = _start_timing()
                    request_id = get_current_request_id()
                    try:
                        result = await inner(*args, **kwargs)
                        self._fire_tool_call(_build_opts(
                            name, args, kwargs, result, None,
                            start, started_at, request_id, metadata,
                        ))
                        return result
                    except Exception as exc:
                        self._fire_tool_call(_build_opts(
                            name, args, kwargs, None, exc,
                            start, started_at, request_id, metadata,
                        ))
                        raise

                return async_wrapped

            @functools.wraps(inner)
            def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
                started_at, start = _start_timing()
                request_id = get_current_request_id()
                try:
                    result = inner(*args, **kwargs)
                    self._fire_tool_call(_build_opts(
                        name, args, kwargs, result, None,
                        start, started_at, request_id, metadata,
                    ))
                    return result
                except Exception as exc:
                    self._fire_tool_call(_build_opts(
                        name, args, kwargs, None, exc,
                        start, started_at, request_id, metadata,
                    ))
                    raise

            return sync_wrapped

        if fn is None:
            return decorator
        return decorator(fn)


class _SyncToolsAPI:
    """Manual tool-call API exposed via ``client.tools``."""

    __slots__ = ("_client",)

    def __init__(self, client: "_BaseSyncClient") -> None:
        self._client = client

    def record(
        self,
        *,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        duration_ms: int = 0,
        error: Optional[str] = None,
        started_at: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Synchronously POST an agent.tool_call event. Surfaces errors.

        ``tool_name`` is required and validated (raises ``ValueError`` if
        empty or non-string). ``request_id`` falls back to the surrounding
        ``with_context`` block when omitted.
        """
        opts = ToolRecordOptions(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            duration_ms=duration_ms,
            error=error,
            started_at=started_at,
            request_id=request_id if request_id is not None else get_current_request_id(),
            metadata=metadata,
        )
        self._client._send_tool_call_event(opts)


# ---------------------------------------------------------------------------
# Helpers shared by sync and async tool wrappers
# ---------------------------------------------------------------------------

def _start_timing() -> tuple[str, float]:
    """Returns (iso8601_started_at, monotonic_start_seconds)."""
    return datetime.now(tz=timezone.utc).isoformat(), time.monotonic()


def _build_opts(
    name: str,
    args: tuple,
    kwargs: dict,
    result: Any,
    exc: Optional[BaseException],
    start: float,
    started_at: str,
    request_id: Optional[str],
    metadata: Optional[Dict[str, Any]],
) -> ToolRecordOptions:
    duration_ms = int((time.monotonic() - start) * 1000)
    return ToolRecordOptions(
        tool_name=name,
        tool_input=_serialize_args(args, kwargs),
        tool_output=None if exc is not None else result,
        duration_ms=duration_ms,
        error=None if exc is None else (str(exc) if not isinstance(exc, str) else exc),
        started_at=started_at,
        request_id=request_id,
        metadata=metadata,
    )


def _serialize_args(args: tuple, kwargs: dict) -> Any:
    """Mirrors Node's serializeArgs.

    - No args → None
    - Single positional arg, no kwargs → that value directly (avoids ``[obj]``)
    - kwargs but no positional → the kwargs dict
    - Otherwise → ``{"args": [...], "kwargs": {...}}``
    """
    if not args and not kwargs:
        return None
    if len(args) == 1 and not kwargs:
        return args[0]
    if not args and kwargs:
        return dict(kwargs)
    return {"args": list(args), "kwargs": dict(kwargs)}


def _warn_on_client_error(status_code: int, debug: bool) -> None:
    """4xx → unconditional ``warnings.warn``; 5xx → only when debug=True."""
    if 200 <= status_code < 300:
        return
    if 400 <= status_code < 500:
        warnings.warn(
            f"[SignalVault] tool_call rejected with {status_code}. "
            f"Check your api_key and event payload.",
            stacklevel=3,
        )
    elif debug:
        warnings.warn(f"[SignalVault] tool_call event failed: {status_code}")


# ---------------------------------------------------------------------------
# Base async client — shared HTTP logic for OpenAI and Anthropic async clients
# ---------------------------------------------------------------------------

class _BaseAsyncClient:
    """Shared HTTP, config, and audit logic for async SignalVault clients."""

    _provider: str = ""  # overridden by subclasses

    def __init__(self, config: SignalVaultConfig) -> None:
        self._config = config
        self._http = httpx.AsyncClient()

    # -- Resource management -------------------------------------------------

    async def aclose(self) -> None:
        """Close the underlying async HTTP connection pool."""
        await self._http.aclose()

    async def __aenter__(self) -> "_BaseAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    # -- Internal HTTP -------------------------------------------------------

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }

    async def _send_request(self, request_id: str, params: dict, metadata: dict) -> Decision:
        try:
            resp = await self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.preflight_timeout,
                json={
                    "type": "ai.request",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": self._provider,
                    "model": params.get("model", ""),
                    "metadata": metadata,
                    "payload": {"messages": params.get("messages", [])},
                },
            )
            if resp.status_code == 200:
                return _parse_decision(resp.json())
        except (httpx.TimeoutException, httpx.RequestError):
            if self._config.debug:
                traceback.print_exc()
        return Decision()

    async def _send_response_from_parts(
        self, request_id: str, model: str, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict,
    ) -> None:
        try:
            await self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.timeout,
                json={
                    "type": "ai.response",
                    "request_id": request_id,
                    "environment": self._config.environment,
                    "provider": self._provider,
                    "model": model,
                    "metadata": metadata,
                    "payload": {
                        "output": output,
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        },
                    },
                },
            )
        except Exception:
            if self._config.debug:
                traceback.print_exc()

    async def _send_audit_from_parts(
        self, request_id: str, model: str, messages: list, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict,
    ) -> None:
        url = f"{self._config.base_url}/v1/events"
        base = {
            "request_id": request_id,
            "environment": self._config.environment,
            "provider": self._provider,
            "model": model,
            "metadata": metadata,
        }
        # Send ai.request FIRST, wait for it, then send ai.response
        try:
            await self._http.post(url, headers=self._headers(), timeout=self._config.timeout, json={
                **base,
                "type": "ai.request",
                "payload": {"messages": messages, "monitor_mode": True},
            })
        except Exception:
            if self._config.debug:
                traceback.print_exc()

        try:
            await self._http.post(url, headers=self._headers(), timeout=self._config.timeout, json={
                **base,
                "type": "ai.response",
                "payload": {
                    "output": output,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    "monitor_mode": True,
                },
            })
        except Exception:
            if self._config.debug:
                traceback.print_exc()

    # -- Agent tool-use capture ---------------------------------------------

    async def _send_tool_call_event(self, opts: ToolRecordOptions) -> None:
        """Awaitable POST of an agent.tool_call event. Surfaces 4xx via warnings."""
        body = build_tool_call_body(
            environment=self._config.environment,
            default_metadata=self._config.metadata,
            opts=opts,
        )
        try:
            resp = await self._http.post(
                f"{self._config.base_url}/v1/events",
                headers=self._headers(),
                timeout=self._config.timeout,
                json=body,
            )
            _warn_on_client_error(resp.status_code, self._config.debug)
        except (httpx.TimeoutException, httpx.RequestError):
            if self._config.debug:
                traceback.print_exc()

    def _fire_tool_call(self, opts: ToolRecordOptions) -> None:
        """Schedule a tool_call audit on the running event loop. Non-blocking.

        If no event loop is running (called from sync code) we silently drop —
        async clients are expected to be used from async code.
        """
        try:
            asyncio.create_task(self._send_tool_call_event(opts))
        except RuntimeError:
            # Loop closed or not running — best-effort.
            pass

    @property
    def tools(self) -> "_AsyncToolsAPI":
        """Manual async tool-call recording API. ``await client.tools.record(...)``."""
        return _AsyncToolsAPI(self)

    def with_context(self, *, request_id: str) -> ToolContext:
        """Returns a context manager scoping ``request_id`` for tool correlation.

        Use as ``async with client.with_context(request_id=...):`` from async
        code.
        """
        return ToolContext(request_id)

    def tool(
        self,
        name: str,
        fn: Optional[Callable[..., Awaitable[Any]]] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable[..., Any]:
        """Wraps an async (or sync) callable so each call records an agent.tool_call.

        The audit POST is fire-and-forget via ``asyncio.create_task`` so the
        wrapped function's latency isn't affected by SignalVault's network call.

        If ``inner`` is a sync function, it is invoked inline inside the async
        wrapper and will briefly block the event loop. For CPU-bound or
        long-running sync work prefer ``run_in_executor`` or use the sync
        :class:`SignalVaultClient`.
        """

        def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
            if inspect.iscoroutinefunction(inner):
                @functools.wraps(inner)
                async def async_wrapped(*args: Any, **kwargs: Any) -> Any:
                    started_at, start = _start_timing()
                    request_id = get_current_request_id()
                    try:
                        result = await inner(*args, **kwargs)
                        self._fire_tool_call(_build_opts(
                            name, args, kwargs, result, None,
                            start, started_at, request_id, metadata,
                        ))
                        return result
                    except Exception as exc:
                        self._fire_tool_call(_build_opts(
                            name, args, kwargs, None, exc,
                            start, started_at, request_id, metadata,
                        ))
                        raise

                return async_wrapped

            # Sync fn through async client — wrap it as async so the wrapper
            # signature is uniform for callers of an AsyncSignalVaultClient.
            @functools.wraps(inner)
            async def sync_wrapped(*args: Any, **kwargs: Any) -> Any:
                started_at, start = _start_timing()
                request_id = get_current_request_id()
                try:
                    result = inner(*args, **kwargs)
                    self._fire_tool_call(_build_opts(
                        name, args, kwargs, result, None,
                        start, started_at, request_id, metadata,
                    ))
                    return result
                except Exception as exc:
                    self._fire_tool_call(_build_opts(
                        name, args, kwargs, None, exc,
                        start, started_at, request_id, metadata,
                    ))
                    raise

            return sync_wrapped

        if fn is None:
            return decorator
        return decorator(fn)


class _AsyncToolsAPI:
    """Async manual tool-call API exposed via ``client.tools``."""

    __slots__ = ("_client",)

    def __init__(self, client: "_BaseAsyncClient") -> None:
        self._client = client

    async def record(
        self,
        *,
        tool_name: str,
        tool_input: Any = None,
        tool_output: Any = None,
        duration_ms: int = 0,
        error: Optional[str] = None,
        started_at: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Awaitable manual recording. Surfaces errors back to the caller."""
        opts = ToolRecordOptions(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            duration_ms=duration_ms,
            error=error,
            started_at=started_at,
            request_id=request_id if request_id is not None else get_current_request_id(),
            metadata=metadata,
        )
        await self._client._send_tool_call_event(opts)


# ---------------------------------------------------------------------------
# SignalVaultClient (sync, OpenAI)
# ---------------------------------------------------------------------------

class _ChatCompletions:
    """Proxies `client.chat.completions.create(...)` with SignalVault guardrails."""

    def __init__(self, sv: "SignalVaultClient"):
        self._sv = sv

    def create(self, **kwargs: Any) -> Any:
        request_id = str(uuid.uuid4())
        sv_metadata = kwargs.pop("sv_metadata", None)
        if sv_metadata is None and "metadata" in kwargs:
            warnings.warn(
                "[SignalVault] 'metadata' kwarg is deprecated, use 'sv_metadata'",
                DeprecationWarning, stacklevel=2,
            )
            sv_metadata = kwargs.pop("metadata", None)
        metadata = _merge_metadata(self._sv._config.metadata, sv_metadata)
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return self._mirror(request_id, kwargs, metadata, stream)
        return self._normal(request_id, kwargs, metadata, stream)

    def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        decision = self._sv._send_request(request_id, kwargs, metadata)

        if decision.decision == "block":
            raise RuntimeError(
                f"[SignalVault] Request blocked: {[v.__dict__ for v in decision.violations]}"
            )
        if decision.decision == "warn" and self._sv._config.debug:
            warnings.warn(f"[SignalVault] Warnings: {decision.violations}")

        if stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        response = self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=False)

        self._sv._fire_response(
            request_id, kwargs.get("model", ""),
            (response.choices[0].message.content or "") if response.choices else "",
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
            metadata,
        )
        return response

    def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        if stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        response = self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=True)

        self._sv._fire_audit(
            request_id, kwargs.get("model", ""), kwargs.get("messages", []),
            (response.choices[0].message.content or "") if response.choices else "",
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
            metadata,
        )
        return response

    def _wrap_stream(
        self, request_id: str, kwargs: dict, stream: Any,
        metadata: dict, mirror: bool,
    ) -> Generator[Any, None, None]:
        chunks: List[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        try:
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    chunks.append(delta.content)
                if hasattr(chunk, "usage") and chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0
                yield chunk
        finally:
            output = "".join(chunks)
            model = kwargs.get("model", "")
            if mirror:
                self._sv._fire_audit(
                    request_id, model, kwargs.get("messages", []),
                    output, prompt_tokens, completion_tokens, metadata,
                )
            else:
                self._sv._fire_response(
                    request_id, model, output, prompt_tokens, completion_tokens, metadata,
                )


class _Chat:
    def __init__(self, sv: "SignalVaultClient"):
        self.completions = _ChatCompletions(sv)


class SignalVaultClient(_BaseSyncClient):
    """
    Sync OpenAI wrapper with SignalVault guardrails.

    Usage::

        from signalvault import SignalVaultClient

        client = SignalVaultClient(
            api_key="sk_live_...",
            openai_api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.signalvault.io",
            metadata={"user_id": "u_123"},
        )

        # Non-streaming
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            sv_metadata={"tool": "clip_detect", "job_id": "abc-123"},
        )

        # Streaming
        for chunk in client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
            sv_metadata={"tool": "stream_test"},
        ):
            print(chunk.choices[0].delta.content or "", end="", flush=True)

        # Context manager (recommended for long-lived use)
        with SignalVaultClient(...) as client:
            response = client.chat.completions.create(...)
    """

    _provider = "openai"

    def __init__(
        self,
        api_key: str,
        openai_api_key: str,
        base_url: str = "http://localhost:4000",
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
        preflight_timeout: float = 2.0,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        from openai import OpenAI
        super().__init__(SignalVaultConfig(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            environment=environment,
            debug=debug,
            mirror_mode=mirror_mode,
            preflight_timeout=preflight_timeout,
            timeout=timeout,
            metadata=metadata or {},
        ))
        self._openai = OpenAI(api_key=openai_api_key)
        self.chat = _Chat(self)


# ---------------------------------------------------------------------------
# AsyncSignalVaultClient (async, OpenAI)
# ---------------------------------------------------------------------------

class _AsyncChatCompletions:
    def __init__(self, sv: "AsyncSignalVaultClient"):
        self._sv = sv

    async def create(self, **kwargs: Any) -> Any:
        request_id = str(uuid.uuid4())
        sv_metadata = kwargs.pop("sv_metadata", None)
        if sv_metadata is None and "metadata" in kwargs:
            warnings.warn(
                "[SignalVault] 'metadata' kwarg is deprecated, use 'sv_metadata'",
                DeprecationWarning, stacklevel=2,
            )
            sv_metadata = kwargs.pop("metadata", None)
        metadata = _merge_metadata(self._sv._config.metadata, sv_metadata)
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return await self._mirror(request_id, kwargs, metadata, stream)
        return await self._normal(request_id, kwargs, metadata, stream)

    async def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        decision = await self._sv._send_request(request_id, kwargs, metadata)

        if decision.decision == "block":
            raise RuntimeError(
                f"[SignalVault] Request blocked: {[v.__dict__ for v in decision.violations]}"
            )
        if decision.decision == "warn" and self._sv._config.debug:
            warnings.warn(f"[SignalVault] Warnings: {decision.violations}")

        if stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        response = await self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=False)

        asyncio.create_task(
            self._sv._send_response_from_parts(
                request_id, kwargs.get("model", ""),
                (response.choices[0].message.content or "") if response.choices else "",
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0,
                metadata,
            )
        )
        return response

    async def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        if stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        response = await self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=True)

        asyncio.create_task(
            self._sv._send_audit_from_parts(
                request_id, kwargs.get("model", ""), kwargs.get("messages", []),
                (response.choices[0].message.content or "") if response.choices else "",
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0,
                metadata,
            )
        )
        return response

    async def _wrap_stream(
        self, request_id: str, kwargs: dict, stream: Any,
        metadata: dict, mirror: bool,
    ) -> AsyncGenerator[Any, None]:
        chunks: List[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    chunks.append(delta.content)
                if hasattr(chunk, "usage") and chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0
                yield chunk
        finally:
            output = "".join(chunks)
            model = kwargs.get("model", "")
            coro = (
                self._sv._send_audit_from_parts(
                    request_id, model, kwargs.get("messages", []),
                    output, prompt_tokens, completion_tokens, metadata,
                )
                if mirror else
                self._sv._send_response_from_parts(
                    request_id, model, output, prompt_tokens, completion_tokens, metadata,
                )
            )
            try:
                asyncio.create_task(coro)
            except RuntimeError:
                pass  # event loop already closed — best-effort


class _AsyncChat:
    def __init__(self, sv: "AsyncSignalVaultClient"):
        self.completions = _AsyncChatCompletions(sv)


class AsyncSignalVaultClient(_BaseAsyncClient):
    """
    Async OpenAI wrapper with SignalVault guardrails. Use in FastAPI, async Django, etc.

    Usage::

        from signalvault import AsyncSignalVaultClient

        client = AsyncSignalVaultClient(
            api_key="sk_live_...",
            openai_api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.signalvault.io",
        )

        # Non-streaming
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello!"}],
            sv_metadata={"tool": "clip_detect"},
        )

        # Streaming
        async for chunk in await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Count to 3"}],
            stream=True,
        ):
            print(chunk.choices[0].delta.content or "", end="", flush=True)

        # Context manager (recommended)
        async with AsyncSignalVaultClient(...) as client:
            response = await client.chat.completions.create(...)
    """

    _provider = "openai"

    def __init__(
        self,
        api_key: str,
        openai_api_key: str,
        base_url: str = "http://localhost:4000",
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
        preflight_timeout: float = 2.0,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        from openai import AsyncOpenAI
        super().__init__(SignalVaultConfig(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            environment=environment,
            debug=debug,
            mirror_mode=mirror_mode,
            preflight_timeout=preflight_timeout,
            timeout=timeout,
            metadata=metadata or {},
        ))
        self._openai = AsyncOpenAI(api_key=openai_api_key)
        self.chat = _AsyncChat(self)


# ---------------------------------------------------------------------------
# AnthropicSignalVaultClient (sync)
# ---------------------------------------------------------------------------

class _AnthropicMessages:
    def __init__(self, sv: "AnthropicSignalVaultClient"):
        self._sv = sv

    def create(self, **kwargs: Any) -> Any:
        request_id = str(uuid.uuid4())
        sv_metadata = kwargs.pop("sv_metadata", None)
        if sv_metadata is None and "metadata" in kwargs:
            warnings.warn(
                "[SignalVault] 'metadata' kwarg is deprecated, use 'sv_metadata'",
                DeprecationWarning, stacklevel=2,
            )
            sv_metadata = kwargs.pop("metadata", None)
        metadata = _merge_metadata(self._sv._config.metadata, sv_metadata)
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return self._mirror(request_id, kwargs, metadata, stream)
        return self._normal(request_id, kwargs, metadata, stream)

    def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        decision = self._sv._send_request(request_id, kwargs, metadata)

        if decision.decision == "block":
            raise RuntimeError(
                f"[SignalVault] Request blocked: {[v.__dict__ for v in decision.violations]}"
            )
        if decision.decision == "warn" and self._sv._config.debug:
            warnings.warn(f"[SignalVault] Warnings: {decision.violations}")

        if stream:
            # Use the streaming context manager to get a proper event iterator
            kwargs.pop("stream", None)
            stream_ctx = self._sv._anthropic.messages.stream(**kwargs)
            return self._wrap_stream(request_id, kwargs, stream_ctx, metadata, mirror=False)

        response = self._sv._anthropic.messages.create(**kwargs)
        self._sv._fire_response(
            request_id, kwargs.get("model", ""),
            response.content[0].text if response.content else "",
            response.usage.input_tokens if response.usage else 0,
            response.usage.output_tokens if response.usage else 0,
            metadata,
        )
        return response

    def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        if stream:
            kwargs.pop("stream", None)
            stream_ctx = self._sv._anthropic.messages.stream(**kwargs)
            return self._wrap_stream(request_id, kwargs, stream_ctx, metadata, mirror=True)

        response = self._sv._anthropic.messages.create(**kwargs)
        self._sv._fire_audit(
            request_id, kwargs.get("model", ""), kwargs.get("messages", []),
            response.content[0].text if response.content else "",
            response.usage.input_tokens if response.usage else 0,
            response.usage.output_tokens if response.usage else 0,
            metadata,
        )
        return response

    def _wrap_stream(
        self, request_id: str, kwargs: dict, stream_ctx: Any,
        metadata: dict, mirror: bool,
    ) -> Generator[Any, None, None]:
        chunks: List[str] = []
        input_tokens = 0
        output_tokens = 0

        try:
            with stream_ctx as stream:
                for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta" and hasattr(event, "delta"):
                            if getattr(event.delta, "type", None) == "text_delta":
                                chunks.append(getattr(event.delta, "text", "") or "")
                        elif event.type == "message_start" and hasattr(event, "message"):
                            usage = getattr(event.message, "usage", None)
                            if usage:
                                input_tokens = getattr(usage, "input_tokens", 0) or 0
                        elif event.type == "message_delta" and hasattr(event, "usage"):
                            output_tokens = getattr(event.usage, "output_tokens", 0) or 0
                    yield event
        finally:
            output = "".join(chunks)
            model = kwargs.get("model", "")
            if mirror:
                self._sv._fire_audit(
                    request_id, model, kwargs.get("messages", []),
                    output, input_tokens, output_tokens, metadata,
                )
            else:
                self._sv._fire_response(
                    request_id, model, output, input_tokens, output_tokens, metadata,
                )


class AnthropicSignalVaultClient(_BaseSyncClient):
    """
    Sync Anthropic/Claude wrapper with SignalVault guardrails.

    Install: pip install signalvault[anthropic]

    Usage::

        from signalvault import AnthropicSignalVaultClient

        client = AnthropicSignalVaultClient(
            api_key="sk_live_...",
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url="https://api.signalvault.io",
        )

        # Non-streaming
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=1024,
            sv_metadata={"tool": "clip_detect"},
        )

        # Streaming
        for event in client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=1024,
            stream=True,
        ):
            if event.type == "content_block_delta":
                print(event.delta.text or "", end="", flush=True)

        # Context manager (recommended)
        with AnthropicSignalVaultClient(...) as client:
            response = client.messages.create(...)
    """

    _provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        anthropic_api_key: str,
        base_url: str = "http://localhost:4000",
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
        preflight_timeout: float = 2.0,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install signalvault[anthropic]"
            )
        super().__init__(SignalVaultConfig(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            environment=environment,
            debug=debug,
            mirror_mode=mirror_mode,
            preflight_timeout=preflight_timeout,
            timeout=timeout,
            metadata=metadata or {},
        ))
        self._anthropic = _anthropic.Anthropic(api_key=anthropic_api_key)
        self.messages = _AnthropicMessages(self)


# ---------------------------------------------------------------------------
# AsyncAnthropicSignalVaultClient
# ---------------------------------------------------------------------------

class _AsyncAnthropicMessages:
    def __init__(self, sv: "AsyncAnthropicSignalVaultClient"):
        self._sv = sv

    async def create(self, **kwargs: Any) -> Any:
        request_id = str(uuid.uuid4())
        sv_metadata = kwargs.pop("sv_metadata", None)
        if sv_metadata is None and "metadata" in kwargs:
            warnings.warn(
                "[SignalVault] 'metadata' kwarg is deprecated, use 'sv_metadata'",
                DeprecationWarning, stacklevel=2,
            )
            sv_metadata = kwargs.pop("metadata", None)
        metadata = _merge_metadata(self._sv._config.metadata, sv_metadata)
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return await self._mirror(request_id, kwargs, metadata, stream)
        return await self._normal(request_id, kwargs, metadata, stream)

    async def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        decision = await self._sv._send_request(request_id, kwargs, metadata)

        if decision.decision == "block":
            raise RuntimeError(
                f"[SignalVault] Request blocked: {[v.__dict__ for v in decision.violations]}"
            )
        if decision.decision == "warn" and self._sv._config.debug:
            warnings.warn(f"[SignalVault] Warnings: {decision.violations}")

        if stream:
            kwargs.pop("stream", None)
            stream_ctx = self._sv._anthropic.messages.stream(**kwargs)
            return self._wrap_stream(request_id, kwargs, stream_ctx, metadata, mirror=False)

        response = await self._sv._anthropic.messages.create(**kwargs)
        asyncio.create_task(
            self._sv._send_response_from_parts(
                request_id, kwargs.get("model", ""),
                response.content[0].text if response.content else "",
                response.usage.input_tokens if response.usage else 0,
                response.usage.output_tokens if response.usage else 0,
                metadata,
            )
        )
        return response

    async def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        if stream:
            kwargs.pop("stream", None)
            stream_ctx = self._sv._anthropic.messages.stream(**kwargs)
            return self._wrap_stream(request_id, kwargs, stream_ctx, metadata, mirror=True)

        response = await self._sv._anthropic.messages.create(**kwargs)
        asyncio.create_task(
            self._sv._send_audit_from_parts(
                request_id, kwargs.get("model", ""), kwargs.get("messages", []),
                response.content[0].text if response.content else "",
                response.usage.input_tokens if response.usage else 0,
                response.usage.output_tokens if response.usage else 0,
                metadata,
            )
        )
        return response

    async def _wrap_stream(
        self, request_id: str, kwargs: dict, stream_ctx: Any,
        metadata: dict, mirror: bool,
    ) -> AsyncGenerator[Any, None]:
        chunks: List[str] = []
        input_tokens = 0
        output_tokens = 0

        try:
            async with stream_ctx as stream:
                async for event in stream:
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta" and hasattr(event, "delta"):
                            if getattr(event.delta, "type", None) == "text_delta":
                                chunks.append(getattr(event.delta, "text", "") or "")
                        elif event.type == "message_start" and hasattr(event, "message"):
                            usage = getattr(event.message, "usage", None)
                            if usage:
                                input_tokens = getattr(usage, "input_tokens", 0) or 0
                        elif event.type == "message_delta" and hasattr(event, "usage"):
                            output_tokens = getattr(event.usage, "output_tokens", 0) or 0
                    yield event
        finally:
            output = "".join(chunks)
            model = kwargs.get("model", "")
            coro = (
                self._sv._send_audit_from_parts(
                    request_id, model, kwargs.get("messages", []),
                    output, input_tokens, output_tokens, metadata,
                )
                if mirror else
                self._sv._send_response_from_parts(
                    request_id, model, output, input_tokens, output_tokens, metadata,
                )
            )
            try:
                asyncio.create_task(coro)
            except RuntimeError:
                pass  # event loop already closed — best-effort


class AsyncAnthropicSignalVaultClient(_BaseAsyncClient):
    """
    Async Anthropic/Claude wrapper with SignalVault guardrails.

    Install: pip install signalvault[anthropic]

    Usage::

        from signalvault import AsyncAnthropicSignalVaultClient

        client = AsyncAnthropicSignalVaultClient(
            api_key="sk_live_...",
            anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url="https://api.signalvault.io",
        )

        # Non-streaming
        response = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=1024,
            sv_metadata={"tool": "clip_detect"},
        )

        # Streaming
        async for event in await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Count to 3"}],
            max_tokens=1024,
            stream=True,
        ):
            if event.type == "content_block_delta":
                print(event.delta.text or "", end="", flush=True)

        # Context manager (recommended)
        async with AsyncAnthropicSignalVaultClient(...) as client:
            response = await client.messages.create(...)
    """

    _provider = "anthropic"

    def __init__(
        self,
        api_key: str,
        anthropic_api_key: str,
        base_url: str = "http://localhost:4000",
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
        preflight_timeout: float = 2.0,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        try:
            import anthropic as _anthropic
        except ImportError:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install signalvault[anthropic]"
            )
        super().__init__(SignalVaultConfig(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            environment=environment,
            debug=debug,
            mirror_mode=mirror_mode,
            preflight_timeout=preflight_timeout,
            timeout=timeout,
            metadata=metadata or {},
        ))
        self._anthropic = _anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self.messages = _AsyncAnthropicMessages(self)
