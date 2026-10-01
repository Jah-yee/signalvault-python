"""SignalVault client — wraps OpenAI and Anthropic with guardrails and audit logging."""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import functools
import inspect
import os
import platform
import random
import sys
import threading
import time
import traceback
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Generator, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from . import __version__ as SDK_VERSION
from .tools import (
    ToolContext,
    ToolRecordOptions,
    build_tool_call_body,
    get_current_request_id,
)


DEFAULT_BASE_URL = "https://api.signalvault.io"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
# Longest Retry-After the SDK waits out before retrying a background event once.
_MAX_RETRY_AFTER_SECONDS = 5.0
# A repeated warning is emitted at most once per this window.
_WARN_INTERVAL_SECONDS = 60.0
_DECISIONS = {"allow", "warn", "block", "redact"}
_FAIL_MODES = {"open", "closed"}
# Pre-flight failures where re-sending the ai.request later can succeed.
_RECORDABLE_CAUSES = {"timeout", "network", "5xx", "invalid-response"}
# Passed as the fallback when no audit events should be sent at all.
_NO_AUDIT: Any = object()
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _emit_warning(message: str, category: type = UserWarning) -> None:
    """Warns from the first stack frame outside this package, every time.

    ``warnings.warn`` would report from a line inside the SDK, and Python's
    default filter shows a given message from a given line only once per
    process — so a second outage hours later would be silent. A fresh
    registry per call avoids that; callers rate-limit instead.
    """
    frame = sys._getframe(1)
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE_DIR + os.sep):
        frame = frame.f_back
    if frame is None:
        filename, lineno, module = "<signalvault>", 0, "signalvault"
    else:
        filename, lineno = frame.f_code.co_filename, frame.f_lineno
        module = frame.f_globals.get("__name__", "")
    warnings.warn_explicit(message, category, filename, lineno, module=module, registry={})


# ---------------------------------------------------------------------------
# Config & shared types
# ---------------------------------------------------------------------------

@dataclass
class SignalVaultConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    environment: str = "production"
    debug: bool = False
    mirror_mode: bool = False
    # "open": allow and warn when no guardrail decision can be obtained.
    # "closed": raise SignalVaultUnavailableError instead.
    fail_mode: str = "open"
    # Timeout for pre-flight /v1/events call (critical path).
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
    # [{"type": ..., "count": ...}] for rules whose redact action matched.
    # Redaction applies to what SignalVault stores; the request sent to the
    # provider is not modified. Check len(), never truthiness.
    redactions: List[Dict[str, Any]] = field(default_factory=list)
    dashboard_url: Optional[str] = None
    # Set when no decision was obtained: "record" if the server may not have
    # stored the request (network, timeout, 5xx, invalid response), "skip" if
    # it refused it (401/402/403/429/3xx). Internal.
    _preflight_failed: Optional[str] = field(default=None, repr=False, compare=False)


class SignalVaultWarning(UserWarning):
    """Emitted when SignalVault cannot check or record a request."""


class SignalVaultBlockedError(RuntimeError):
    """Raised when a guardrail rule blocks the request.

    Subclasses ``RuntimeError`` so existing ``except RuntimeError`` handlers
    keep working.
    """

    def __init__(
        self, request_id: str, violations: List[Violation], dashboard_url: Optional[str] = None,
    ) -> None:
        types = ", ".join(dict.fromkeys(str(v.type) for v in violations if v.type)) or "policy"
        super().__init__(f"[SignalVault] Request blocked by guardrails ({types}).")
        self.request_id = request_id
        self.violations = violations
        self.dashboard_url = dashboard_url


class SignalVaultUnavailableError(RuntimeError):
    """Raised with ``fail_mode="closed"`` when no guardrail decision could be obtained."""

    def __init__(self, request_id: str, reason: str, status: Optional[int] = None) -> None:
        super().__init__(
            f"[SignalVault] Guardrail check unavailable: {reason}. "
            f"Request not sent (fail_mode='closed')."
        )
        self.request_id = request_id
        self.status = status


# ---------------------------------------------------------------------------
# Module-level persistent background executor
# ---------------------------------------------------------------------------

_BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sv-audit")
atexit.register(_BACKGROUND_EXECUTOR.shutdown, wait=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_base_url(raw: str) -> str:
    """Strips trailing slashes and refuses plaintext HTTP except to localhost.

    The API key and full prompts travel in every request, so an ``http://``
    URL would expose both before the server could redirect to HTTPS.
    """
    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"[SignalVault] base_url must be an https:// URL, got {raw!r}")
    if parsed.query or parsed.fragment:
        raise ValueError(f"[SignalVault] base_url must not contain a query or fragment, got {raw!r}")
    if parsed.scheme == "http" and parsed.hostname not in _LOCAL_HOSTS:
        raise ValueError(
            "[SignalVault] base_url must use https:// (plain http:// is only allowed for "
            "localhost). Your API key and prompts would otherwise be sent unencrypted."
        )
    return raw.rstrip("/")


def _build_config(
    api_key: str, base_url: str, environment: str, debug: bool, mirror_mode: bool,
    fail_mode: str, preflight_timeout: float, timeout: float,
    metadata: Optional[Dict[str, Any]],
) -> SignalVaultConfig:
    if fail_mode not in _FAIL_MODES:
        raise ValueError("[SignalVault] fail_mode must be 'open' or 'closed'")
    return SignalVaultConfig(
        api_key=api_key,
        base_url=normalize_base_url(base_url),
        environment=environment,
        debug=debug,
        mirror_mode=mirror_mode,
        fail_mode=fail_mode,
        preflight_timeout=preflight_timeout,
        timeout=timeout,
        metadata=metadata or {},
    )


def _merge_metadata(config_meta: Dict[str, Any], call_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {**config_meta, **(call_meta or {})}


def _parse_decision(data: dict) -> Decision:
    known = Violation.__dataclass_fields__
    violations = [
        Violation(**{k: v for k, v in item.items() if k in known})
        for item in (data.get("violations") if isinstance(data.get("violations"), list) else [])
        if isinstance(item, dict)
    ]
    redactions = data.get("redactions")
    dashboard_url = data.get("dashboard_url")
    return Decision(
        decision=data.get("decision", "allow"),
        violations=violations,
        redactions=redactions if isinstance(redactions, list) else [],
        dashboard_url=dashboard_url if isinstance(dashboard_url, str) else None,
    )


def _anthropic_messages_for_audit(params: dict) -> list:
    """Includes Anthropic's top-level ``system`` prompt as a system message.

    The server scans ``payload.messages``; without this, secrets or PII in the
    system prompt go unchecked.
    """
    messages = list(params.get("messages") or [])
    system = params.get("system")
    if system in (None, "", []):
        return messages
    return [{"role": "system", "content": system}, *messages]


def _anthropic_output_text(response: Any) -> str:
    """Concatenates every text block; ``content[0]`` alone misses text after a tool_use block."""
    blocks = getattr(response, "content", None) or []
    return "".join(
        getattr(b, "text", "") or ""
        for b in blocks
        if getattr(b, "type", "text") == "text" and isinstance(getattr(b, "text", None), str)
    )


def _retry_delay(resp: Optional[httpx.Response]) -> Optional[float]:
    """Seconds to wait before retrying a failed response once, or None for no retry."""
    if resp is None:
        return None
    if resp.status_code == 429:
        try:
            retry_after = float(resp.headers.get("retry-after"))
        except (TypeError, ValueError):
            return None
        return retry_after if 0 <= retry_after <= _MAX_RETRY_AFTER_SECONDS else None
    if resp.status_code >= 500:
        return 0.25 + random.random() * 0.5
    return None


def _describe_failure(resp: httpx.Response) -> Tuple[str, str]:
    """Maps a non-2xx pre-flight response to (warning key, human reason)."""
    try:
        body = resp.json()
    except Exception:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    error_type = error.get("type") if isinstance(error, dict) else error
    status = resp.status_code

    if status == 401:
        return "401", "SignalVault rejected the API key (401: invalid or revoked)"
    if status == 402:
        return "402", f"SignalVault account is not active (402: {error_type or 'payment required'})"
    if status == 403:
        return "403", f"SignalVault denied access (403: {error_type or 'forbidden'})"
    if status == 429:
        if error_type == "trial_limit_exceeded":
            return "429-trial", "SignalVault trial limit reached (429)"
        return "429", "SignalVault rate limit reached (429)"
    if 300 <= status < 400:
        return "3xx", f"SignalVault API redirected ({status}); check that base_url is the https:// API URL"
    if status >= 500:
        return "5xx", f"SignalVault API error ({status})"
    return str(status), f"SignalVault rejected the pre-flight check ({status})"


class _ClientCommon:
    """Config, headers, warnings and decision handling shared by all clients."""

    _provider: str = ""  # overridden by subclasses

    def _init_common(self, config: SignalVaultConfig) -> None:
        self._config = config
        self._last_warned: Dict[str, float] = {}
        self._warn_lock = threading.Lock()

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
            "User-Agent": f"signalvault-python/{SDK_VERSION} python/{platform.python_version()}",
        }

    def _audit_messages(self, params: dict) -> list:
        """Messages as sent to SignalVault. Anthropic clients add the system prompt."""
        return list(params.get("messages") or [])

    def _warn(self, key: str, message: str) -> None:
        """Warns at most once per minute per key. Not gated on debug."""
        now = time.monotonic()
        with self._warn_lock:
            last = self._last_warned.get(key)
            if last is not None and now - last < _WARN_INTERVAL_SECONDS:
                return
            self._last_warned[key] = now
        _emit_warning(message, SignalVaultWarning)

    def _unavailable(
        self, request_id: str, key: str, reason: str, status: Optional[int] = None,
    ) -> Decision:
        """No decision could be obtained: raise in fail_mode 'closed', else warn and allow."""
        if self._config.fail_mode == "closed":
            raise SignalVaultUnavailableError(request_id, reason, status)
        self._warn(
            f"preflight:{key}",
            f"[SignalVault] {reason}. Guardrails were NOT applied — requests are being sent "
            f"to the provider unchecked (fail_mode='open').",
        )
        return Decision(_preflight_failed="record" if key in _RECORDABLE_CAUSES else "skip")

    def _fallback_request(self, decision: Decision, request_id: str, params: dict, metadata: dict) -> Any:
        """What to send before the response when the pre-flight got no decision.

        None: the pre-flight was recorded normally. ``_NO_AUDIT``: the server
        refused it, so a response event would be refused too. Otherwise the
        ai.request body, so an unchecked call still reaches the audit log.
        """
        if decision._preflight_failed is None:
            return None
        if decision._preflight_failed == "skip":
            return _NO_AUDIT
        body = self._preflight_body(request_id, params, metadata)
        body["payload"]["preflight_unavailable"] = True
        return body

    def _preflight_body(self, request_id: str, params: dict, metadata: dict) -> dict:
        return {
            "type": "ai.request",
            "request_id": request_id,
            "environment": self._config.environment,
            "provider": self._provider,
            "model": params.get("model", ""),
            "metadata": metadata,
            "payload": {"messages": self._audit_messages(params)},
        }

    def _interpret_preflight(self, request_id: str, resp: httpx.Response) -> Decision:
        if not 200 <= resp.status_code < 300:
            key, reason = _describe_failure(resp)
            return self._unavailable(request_id, key, reason, resp.status_code)
        try:
            data = resp.json()
        except ValueError:
            return self._unavailable(request_id, "invalid-response", "SignalVault returned an invalid response")
        decision = data.get("decision") if isinstance(data, dict) else None
        if not isinstance(decision, str) or decision not in _DECISIONS:
            return self._unavailable(request_id, "invalid-response", "SignalVault returned an invalid decision")
        return _parse_decision(data)

    def _enforce(self, request_id: str, decision: Decision) -> None:
        if decision.decision == "block":
            raise SignalVaultBlockedError(request_id, decision.violations, decision.dashboard_url)
        if decision.decision == "warn" and self._config.debug:
            warnings.warn(f"[SignalVault] Warnings: {decision.violations}", SignalVaultWarning)

    def _response_body(
        self, request_id: str, model: str, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict, monitor_mode: bool = False,
    ) -> dict:
        payload: Dict[str, Any] = {
            "output": output,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
        if monitor_mode:
            payload["monitor_mode"] = True
        return {
            "type": "ai.response",
            "request_id": request_id,
            "environment": self._config.environment,
            "provider": self._provider,
            "model": model,
            "metadata": metadata,
            "payload": payload,
        }

    def _mirror_request_body(self, request_id: str, model: str, messages: list, metadata: dict) -> dict:
        return {
            "type": "ai.request",
            "request_id": request_id,
            "environment": self._config.environment,
            "provider": self._provider,
            "model": model,
            "metadata": metadata,
            "payload": {"messages": messages, "monitor_mode": True},
        }

    def _log_undelivered(self, resp: Optional[httpx.Response]) -> None:
        if resp is not None and resp.status_code == 429:
            self._warn("bg-429", "[SignalVault] Rate limited (429): audit events are being dropped.")
        elif self._config.debug:
            status = resp.status_code if resp is not None else "no response"
            _emit_warning(f"[SignalVault] Event not delivered: {status}", SignalVaultWarning)

    def _prepare_tool_call(self, opts: ToolRecordOptions) -> Optional[dict]:
        """Builds the tool_call body now, so it reflects the arguments at call time.

        Returns None (and warns) instead of raising — a wrapper must never
        break the user's tool because the audit could not be built.
        """
        try:
            return build_tool_call_body(
                environment=self._config.environment,
                default_metadata=self._config.metadata,
                opts=opts,
            )
        except Exception as exc:
            self._warn("tool-build", f"[SignalVault] tool_call event dropped: {exc}")
            return None


# ---------------------------------------------------------------------------
# Base sync client — shared HTTP logic for OpenAI and Anthropic sync clients
# ---------------------------------------------------------------------------

class _BaseSyncClient(_ClientCommon):
    """Shared HTTP, config, and audit logic for sync SignalVault clients."""

    def __init__(self, config: SignalVaultConfig) -> None:
        self._init_common(config)
        # Never follow redirects: the API key must not be re-sent to another URL.
        self._http = httpx.Client(follow_redirects=False)
        self._pending: set = set()
        self._pending_lock = threading.Lock()

    # -- Resource management -------------------------------------------------

    def flush(self, timeout: float = 5.0) -> None:
        """Waits up to ``timeout`` seconds for queued audit events to be sent."""
        with self._pending_lock:
            pending = list(self._pending)
        if not pending:
            return
        _, not_done = concurrent.futures.wait(pending, timeout=timeout)
        if not_done:
            self._warn("flush-timeout", f"[SignalVault] flush() timed out with {len(not_done)} event(s) still sending.")

    def close(self, timeout: float = 5.0) -> None:
        """Sends queued audit events (up to ``timeout``), then closes the HTTP pool."""
        self.flush(timeout)
        self._http.close()

    def __enter__(self) -> "_BaseSyncClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _submit(self, fn: Callable[..., Any], *args: Any) -> None:
        future = _BACKGROUND_EXECUTOR.submit(fn, *args)
        with self._pending_lock:
            self._pending.add(future)

        def _done(f: concurrent.futures.Future) -> None:
            with self._pending_lock:
                self._pending.discard(f)
            if not f.cancelled() and f.exception() is not None and self._config.debug:
                exc = f.exception()
                traceback.print_exception(type(exc), exc, exc.__traceback__)

        future.add_done_callback(_done)

    # -- Internal HTTP -------------------------------------------------------

    def _post(self, body: dict, timeout: float) -> httpx.Response:
        return self._http.post(
            f"{self._config.base_url}/v1/events",
            headers=self._headers(),
            timeout=timeout,
            json=body,
        )

    def _post_background(
        self, body: dict, *, event_id: bool = True, retry: bool = True,
    ) -> Optional[httpx.Response]:
        """POST with one retry on connection errors, 5xx, and 429 with a short Retry-After.

        ``event_id`` adds an idempotency key so a retry is never double-counted;
        it is off only for the fallback ai.request, which relies on the
        server's request_id de-duplication (the original pre-flight may have
        been stored before the client gave up). Timeouts are not retried: the
        server may be slow rather than down, and a retry would hold a worker
        twice as long. Returns the last response, or None.
        """
        event = {"event_id": str(uuid.uuid4()), **body} if event_id else body
        resp: Optional[httpx.Response] = None
        delay: Optional[float] = None
        try:
            resp = self._post(event, self._config.timeout)
            if 200 <= resp.status_code < 300:
                return resp
            delay = _retry_delay(resp)
        except httpx.TimeoutException:
            if self._config.debug:
                traceback.print_exc()
        except httpx.HTTPError:
            if self._config.debug:
                traceback.print_exc()
            delay = 0.25 + random.random() * 0.5

        if retry and delay is not None:
            time.sleep(delay)
            try:
                resp = self._post(event, self._config.timeout)
                if 200 <= resp.status_code < 300:
                    return resp
            except httpx.HTTPError:
                if self._config.debug:
                    traceback.print_exc()

        self._log_undelivered(resp)
        return resp

    def _send_request(self, request_id: str, params: dict, metadata: dict) -> Decision:
        try:
            resp = self._post(
                self._preflight_body(request_id, params, metadata), self._config.preflight_timeout,
            )
        except httpx.TimeoutException:
            return self._unavailable(
                request_id, "timeout",
                f"SignalVault pre-flight check timed out after {self._config.preflight_timeout}s",
            )
        except httpx.HTTPError:
            if self._config.debug:
                traceback.print_exc()
            return self._unavailable(request_id, "network", "SignalVault API is unreachable")
        return self._interpret_preflight(request_id, resp)

    def _fire_response(
        self, request_id: str, model: str, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict, fallback: Any = None,
    ) -> None:
        """Submit response event to background executor — does not block caller."""
        if fallback is _NO_AUDIT:
            return
        self._submit(
            self._send_response_from_parts,
            request_id, model, output, prompt_tokens, completion_tokens, metadata, fallback,
        )

    def _send_response_from_parts(
        self, request_id: str, model: str, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict, fallback: Any = None,
    ) -> None:
        if fallback is _NO_AUDIT:
            return
        if fallback is not None:
            # Must land first: the server rejects an ai.response whose ai.request it has not stored.
            self._post_background(fallback, event_id=False)
        self._post_background(self._response_body(
            request_id, model, output, prompt_tokens, completion_tokens, metadata,
        ))

    def _fire_audit(
        self, request_id: str, model: str, messages: list, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict,
    ) -> None:
        """Submit audit events to background executor — does not block caller."""
        self._submit(
            self._send_audit_from_parts,
            request_id, model, messages, output, prompt_tokens, completion_tokens, metadata,
        )

    def _send_audit_from_parts(
        self, request_id: str, model: str, messages: list, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict,
    ) -> None:
        """Sequential on purpose: the server rejects an ai.response whose ai.request it has not stored."""
        self._post_background(self._mirror_request_body(request_id, model, messages, metadata))
        self._post_background(self._response_body(
            request_id, model, output, prompt_tokens, completion_tokens, metadata, monitor_mode=True,
        ))

    # -- Agent tool-use capture ---------------------------------------------

    def _send_tool_call_event(self, opts: ToolRecordOptions) -> None:
        """Blocking POST of an agent.tool_call event. Surfaces 4xx via warnings.

        Used by the manual API (``client.tools.record``) directly. Raises
        ``ValueError`` on an invalid ``tool_name``.
        """
        body = build_tool_call_body(
            environment=self._config.environment,
            default_metadata=self._config.metadata,
            opts=opts,
        )
        self._deliver_tool_call(body, retry=False)

    def _deliver_tool_call(self, body: dict, retry: bool = True) -> None:
        resp = self._post_background(body, retry=retry)
        if resp is not None and resp.status_code != 429:
            _warn_on_client_error(resp.status_code, self._config.debug)

    def _fire_tool_call(self, opts: ToolRecordOptions) -> None:
        """Builds the event now and sends it in the background. Non-blocking."""
        body = self._prepare_tool_call(opts)
        if body is not None:
            self._submit(self._deliver_tool_call, body)

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
            SignalVaultWarning,
            stacklevel=3,
        )
    elif debug:
        warnings.warn(f"[SignalVault] tool_call event failed: {status_code}", SignalVaultWarning)


# ---------------------------------------------------------------------------
# Base async client — shared HTTP logic for OpenAI and Anthropic async clients
# ---------------------------------------------------------------------------

class _BaseAsyncClient(_ClientCommon):
    """Shared HTTP, config, and audit logic for async SignalVault clients."""

    def __init__(self, config: SignalVaultConfig) -> None:
        self._init_common(config)
        # Never follow redirects: the API key must not be re-sent to another URL.
        self._http = httpx.AsyncClient(follow_redirects=False)
        # Strong references: the event loop keeps only weak references to
        # tasks, so an unreferenced audit task can be garbage-collected mid-send.
        self._tasks: set = set()

    # -- Resource management -------------------------------------------------

    async def flush(self, timeout: float = 5.0) -> None:
        """Waits up to ``timeout`` seconds for queued audit events to be sent."""
        tasks = [t for t in self._tasks if not t.done()]
        if not tasks:
            return
        _, pending = await asyncio.wait(tasks, timeout=timeout)
        if pending:
            self._warn("flush-timeout", f"[SignalVault] flush() timed out with {len(pending)} event(s) still sending.")

    async def aclose(self, timeout: float = 5.0) -> None:
        """Sends queued audit events (up to ``timeout``), then closes the HTTP pool."""
        await self.flush(timeout)
        await self._http.aclose()

    async def __aenter__(self) -> "_BaseAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def _spawn(self, coro: Awaitable[Any]) -> None:
        """Runs ``coro`` in the background, keeping a reference until it finishes."""
        try:
            task = asyncio.get_running_loop().create_task(coro)  # type: ignore[arg-type]
        except RuntimeError:
            coro.close()  # type: ignore[attr-defined]
            self._warn(
                "no-loop",
                "[SignalVault] No running event loop: audit event dropped. "
                "Use the async client from async code.",
            )
            return
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: "asyncio.Task[Any]") -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None and self._config.debug:
            exc = task.exception()
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    # -- Internal HTTP -------------------------------------------------------

    async def _post(self, body: dict, timeout: float) -> httpx.Response:
        return await self._http.post(
            f"{self._config.base_url}/v1/events",
            headers=self._headers(),
            timeout=timeout,
            json=body,
        )

    async def _post_background(
        self, body: dict, *, event_id: bool = True, retry: bool = True,
    ) -> Optional[httpx.Response]:
        """POST with one retry on connection errors, 5xx, and 429 with a short Retry-After.

        ``event_id`` adds an idempotency key so a retry is never double-counted;
        it is off only for the fallback ai.request, which relies on the
        server's request_id de-duplication (the original pre-flight may have
        been stored before the client gave up). Timeouts are not retried: the
        server may be slow rather than down, and a retry would hold a worker
        twice as long. Returns the last response, or None.
        """
        event = {"event_id": str(uuid.uuid4()), **body} if event_id else body
        resp: Optional[httpx.Response] = None
        delay: Optional[float] = None
        try:
            resp = await self._post(event, self._config.timeout)
            if 200 <= resp.status_code < 300:
                return resp
            delay = _retry_delay(resp)
        except httpx.TimeoutException:
            if self._config.debug:
                traceback.print_exc()
        except httpx.HTTPError:
            if self._config.debug:
                traceback.print_exc()
            delay = 0.25 + random.random() * 0.5

        if retry and delay is not None:
            await asyncio.sleep(delay)
            try:
                resp = await self._post(event, self._config.timeout)
                if 200 <= resp.status_code < 300:
                    return resp
            except httpx.HTTPError:
                if self._config.debug:
                    traceback.print_exc()

        self._log_undelivered(resp)
        return resp

    async def _send_request(self, request_id: str, params: dict, metadata: dict) -> Decision:
        try:
            resp = await self._post(
                self._preflight_body(request_id, params, metadata), self._config.preflight_timeout,
            )
        except httpx.TimeoutException:
            return self._unavailable(
                request_id, "timeout",
                f"SignalVault pre-flight check timed out after {self._config.preflight_timeout}s",
            )
        except httpx.HTTPError:
            if self._config.debug:
                traceback.print_exc()
            return self._unavailable(request_id, "network", "SignalVault API is unreachable")
        return self._interpret_preflight(request_id, resp)

    async def _send_response_from_parts(
        self, request_id: str, model: str, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict, fallback: Any = None,
    ) -> None:
        if fallback is _NO_AUDIT:
            return
        if fallback is not None:
            # Must land first: the server rejects an ai.response whose ai.request it has not stored.
            await self._post_background(fallback, event_id=False)
        await self._post_background(self._response_body(
            request_id, model, output, prompt_tokens, completion_tokens, metadata,
        ))

    async def _send_audit_from_parts(
        self, request_id: str, model: str, messages: list, output: str,
        prompt_tokens: int, completion_tokens: int, metadata: dict,
    ) -> None:
        """Sequential on purpose: the server rejects an ai.response whose ai.request it has not stored."""
        await self._post_background(self._mirror_request_body(request_id, model, messages, metadata))
        await self._post_background(self._response_body(
            request_id, model, output, prompt_tokens, completion_tokens, metadata, monitor_mode=True,
        ))

    # -- Agent tool-use capture ---------------------------------------------

    async def _send_tool_call_event(self, opts: ToolRecordOptions) -> None:
        """Awaitable POST of an agent.tool_call event. Surfaces 4xx via warnings."""
        body = build_tool_call_body(
            environment=self._config.environment,
            default_metadata=self._config.metadata,
            opts=opts,
        )
        await self._deliver_tool_call(body, retry=False)

    async def _deliver_tool_call(self, body: dict, retry: bool = True) -> None:
        resp = await self._post_background(body, retry=retry)
        if resp is not None and resp.status_code != 429:
            _warn_on_client_error(resp.status_code, self._config.debug)

    def _fire_tool_call(self, opts: ToolRecordOptions) -> None:
        """Builds the event now and schedules the send on the running loop. Non-blocking."""
        body = self._prepare_tool_call(opts)
        if body is not None:
            self._spawn(self._deliver_tool_call(body))

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

        The audit POST is fire-and-forget on the running event loop so the
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
        # `metadata` belongs to the provider (e.g. Anthropic's user_id) and is
        # passed through untouched; SignalVault metadata goes in `sv_metadata`.
        sv_metadata = kwargs.pop("sv_metadata", None)
        metadata = _merge_metadata(self._sv._config.metadata, sv_metadata)
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return self._mirror(request_id, kwargs, metadata, stream)
        return self._normal(request_id, kwargs, metadata, stream)

    def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        decision = self._sv._send_request(request_id, kwargs, metadata)

        self._sv._enforce(request_id, decision)
        fallback = self._sv._fallback_request(decision, request_id, kwargs, metadata)

        if stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        response = self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=False, fallback=fallback)

        self._sv._fire_response(
            request_id, kwargs.get("model", ""),
            (response.choices[0].message.content or "") if response.choices else "",
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
            metadata, fallback=fallback,
        )
        return response

    def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        if stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        response = self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=True)

        self._sv._fire_audit(
            request_id, kwargs.get("model", ""), self._sv._audit_messages(kwargs),
            (response.choices[0].message.content or "") if response.choices else "",
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
            metadata,
        )
        return response

    def _wrap_stream(
        self, request_id: str, kwargs: dict, stream: Any,
        metadata: dict, mirror: bool, fallback: Any = None,
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
                    request_id, model, self._sv._audit_messages(kwargs),
                    output, prompt_tokens, completion_tokens, metadata,
                )
            else:
                self._sv._fire_response(
                    request_id, model, output, prompt_tokens, completion_tokens, metadata, fallback=fallback,
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
        base_url: str = DEFAULT_BASE_URL,
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
        fail_mode: str = "open",
        preflight_timeout: float = 2.0,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        from openai import OpenAI
        super().__init__(_build_config(
            api_key, base_url, environment, debug, mirror_mode,
            fail_mode, preflight_timeout, timeout, metadata,
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
        # `metadata` belongs to the provider (e.g. Anthropic's user_id) and is
        # passed through untouched; SignalVault metadata goes in `sv_metadata`.
        sv_metadata = kwargs.pop("sv_metadata", None)
        metadata = _merge_metadata(self._sv._config.metadata, sv_metadata)
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return await self._mirror(request_id, kwargs, metadata, stream)
        return await self._normal(request_id, kwargs, metadata, stream)

    async def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        decision = await self._sv._send_request(request_id, kwargs, metadata)

        self._sv._enforce(request_id, decision)
        fallback = self._sv._fallback_request(decision, request_id, kwargs, metadata)

        if stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        response = await self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=False, fallback=fallback)

        self._sv._spawn(
            self._sv._send_response_from_parts(
                request_id, kwargs.get("model", ""),
                (response.choices[0].message.content or "") if response.choices else "",
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0,
                metadata, fallback=fallback,
            )
        )
        return response

    async def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        if stream and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}

        response = await self._sv._openai.chat.completions.create(**kwargs)

        if stream:
            return self._wrap_stream(request_id, kwargs, response, metadata, mirror=True)

        self._sv._spawn(
            self._sv._send_audit_from_parts(
                request_id, kwargs.get("model", ""), self._sv._audit_messages(kwargs),
                (response.choices[0].message.content or "") if response.choices else "",
                response.usage.prompt_tokens if response.usage else 0,
                response.usage.completion_tokens if response.usage else 0,
                metadata,
            )
        )
        return response

    async def _wrap_stream(
        self, request_id: str, kwargs: dict, stream: Any,
        metadata: dict, mirror: bool, fallback: Any = None,
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
                    request_id, model, self._sv._audit_messages(kwargs),
                    output, prompt_tokens, completion_tokens, metadata,
                )
                if mirror else
                self._sv._send_response_from_parts(
                    request_id, model, output, prompt_tokens, completion_tokens, metadata, fallback=fallback,
                )
            )
            self._sv._spawn(coro)


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
        base_url: str = DEFAULT_BASE_URL,
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
        fail_mode: str = "open",
        preflight_timeout: float = 2.0,
        timeout: float = 30.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        from openai import AsyncOpenAI
        super().__init__(_build_config(
            api_key, base_url, environment, debug, mirror_mode,
            fail_mode, preflight_timeout, timeout, metadata,
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
        # `metadata` belongs to the provider (e.g. Anthropic's user_id) and is
        # passed through untouched; SignalVault metadata goes in `sv_metadata`.
        sv_metadata = kwargs.pop("sv_metadata", None)
        metadata = _merge_metadata(self._sv._config.metadata, sv_metadata)
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return self._mirror(request_id, kwargs, metadata, stream)
        return self._normal(request_id, kwargs, metadata, stream)

    def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        decision = self._sv._send_request(request_id, kwargs, metadata)

        self._sv._enforce(request_id, decision)
        fallback = self._sv._fallback_request(decision, request_id, kwargs, metadata)

        if stream:
            # Use the streaming context manager to get a proper event iterator
            kwargs.pop("stream", None)
            stream_ctx = self._sv._anthropic.messages.stream(**kwargs)
            return self._wrap_stream(request_id, kwargs, stream_ctx, metadata, mirror=False, fallback=fallback)

        response = self._sv._anthropic.messages.create(**kwargs)
        self._sv._fire_response(
            request_id, kwargs.get("model", ""),
            _anthropic_output_text(response),
            response.usage.input_tokens if response.usage else 0,
            response.usage.output_tokens if response.usage else 0,
            metadata, fallback=fallback,
        )
        return response

    def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        if stream:
            kwargs.pop("stream", None)
            stream_ctx = self._sv._anthropic.messages.stream(**kwargs)
            return self._wrap_stream(request_id, kwargs, stream_ctx, metadata, mirror=True)

        response = self._sv._anthropic.messages.create(**kwargs)
        self._sv._fire_audit(
            request_id, kwargs.get("model", ""), self._sv._audit_messages(kwargs),
            _anthropic_output_text(response),
            response.usage.input_tokens if response.usage else 0,
            response.usage.output_tokens if response.usage else 0,
            metadata,
        )
        return response

    def _wrap_stream(
        self, request_id: str, kwargs: dict, stream_ctx: Any,
        metadata: dict, mirror: bool, fallback: Any = None,
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
                    request_id, model, self._sv._audit_messages(kwargs),
                    output, input_tokens, output_tokens, metadata,
                )
            else:
                self._sv._fire_response(
                    request_id, model, output, input_tokens, output_tokens, metadata, fallback=fallback,
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

    def _audit_messages(self, params: dict) -> list:
        return _anthropic_messages_for_audit(params)

    def __init__(
        self,
        api_key: str,
        anthropic_api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
        fail_mode: str = "open",
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
        super().__init__(_build_config(
            api_key, base_url, environment, debug, mirror_mode,
            fail_mode, preflight_timeout, timeout, metadata,
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
        # `metadata` belongs to the provider (e.g. Anthropic's user_id) and is
        # passed through untouched; SignalVault metadata goes in `sv_metadata`.
        sv_metadata = kwargs.pop("sv_metadata", None)
        metadata = _merge_metadata(self._sv._config.metadata, sv_metadata)
        stream = kwargs.get("stream", False)

        if self._sv._config.mirror_mode:
            return await self._mirror(request_id, kwargs, metadata, stream)
        return await self._normal(request_id, kwargs, metadata, stream)

    async def _normal(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        decision = await self._sv._send_request(request_id, kwargs, metadata)

        self._sv._enforce(request_id, decision)
        fallback = self._sv._fallback_request(decision, request_id, kwargs, metadata)

        if stream:
            kwargs.pop("stream", None)
            stream_ctx = self._sv._anthropic.messages.stream(**kwargs)
            return self._wrap_stream(request_id, kwargs, stream_ctx, metadata, mirror=False, fallback=fallback)

        response = await self._sv._anthropic.messages.create(**kwargs)
        self._sv._spawn(
            self._sv._send_response_from_parts(
                request_id, kwargs.get("model", ""),
                _anthropic_output_text(response),
                response.usage.input_tokens if response.usage else 0,
                response.usage.output_tokens if response.usage else 0,
                metadata, fallback=fallback,
            )
        )
        return response

    async def _mirror(self, request_id: str, kwargs: dict, metadata: dict, stream: bool) -> Any:
        if stream:
            kwargs.pop("stream", None)
            stream_ctx = self._sv._anthropic.messages.stream(**kwargs)
            return self._wrap_stream(request_id, kwargs, stream_ctx, metadata, mirror=True)

        response = await self._sv._anthropic.messages.create(**kwargs)
        self._sv._spawn(
            self._sv._send_audit_from_parts(
                request_id, kwargs.get("model", ""), self._sv._audit_messages(kwargs),
                _anthropic_output_text(response),
                response.usage.input_tokens if response.usage else 0,
                response.usage.output_tokens if response.usage else 0,
                metadata,
            )
        )
        return response

    async def _wrap_stream(
        self, request_id: str, kwargs: dict, stream_ctx: Any,
        metadata: dict, mirror: bool, fallback: Any = None,
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
                    request_id, model, self._sv._audit_messages(kwargs),
                    output, input_tokens, output_tokens, metadata,
                )
                if mirror else
                self._sv._send_response_from_parts(
                    request_id, model, output, input_tokens, output_tokens, metadata, fallback=fallback,
                )
            )
            self._sv._spawn(coro)


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

    def _audit_messages(self, params: dict) -> list:
        return _anthropic_messages_for_audit(params)

    def __init__(
        self,
        api_key: str,
        anthropic_api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        environment: str = "production",
        debug: bool = False,
        mirror_mode: bool = False,
        fail_mode: str = "open",
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
        super().__init__(_build_config(
            api_key, base_url, environment, debug, mirror_mode,
            fail_mode, preflight_timeout, timeout, metadata,
        ))
        self._anthropic = _anthropic.AsyncAnthropic(api_key=anthropic_api_key)
        self.messages = _AsyncAnthropicMessages(self)
