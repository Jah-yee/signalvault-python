"""Agent tool-use capture helpers — shared between sync and async clients.

The public surface (mirrored from the Node SDK):

- ``client.tool(name, fn)`` — decorator-style wrapper that records every call.
- ``client.tools.record(...)`` — manual API for streaming / post-hoc capture.
- ``client.with_context(request_id=...)`` — context manager that auto-correlates
  enclosed tool calls to a parent ai.request via a ``contextvars.ContextVar``.

Defensive behavior matches the Node SDK byte-for-byte:

- ``tool_name`` truncated to 200 bytes
- ``error`` truncated to 1900 bytes
- ``tool_input`` / ``tool_output`` sanitized (circular refs replaced with the
  ``"[Circular]"`` marker, non-JSON-serializable values replaced with markers)
  and capped at 256 KB with a truncation envelope.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Byte limits — match server-side validation in TS payload validator
# ---------------------------------------------------------------------------

MAX_PAYLOAD_BYTES: int = 256 * 1024  # 256 KB
MAX_ERROR_BYTES: int = 1900
MAX_TOOL_NAME_BYTES: int = 200

# ---------------------------------------------------------------------------
# Context propagation
# ---------------------------------------------------------------------------

# A single ContextVar serves both sync (with-block) and async (asyncio Task) —
# Python copies the context across asyncio task boundaries automatically.
_request_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "signalvault_request_id", default=None
)


def get_current_request_id() -> Optional[str]:
    """Returns the request_id from the surrounding ``with_context`` block, or None."""
    return _request_id_ctx.get()


class ToolContext:
    """Context manager that scopes a ``request_id`` for tool-call correlation.

    Usable as both a sync context manager (``with``) and async context manager
    (``async with``), so the same ``client.with_context(...)`` call works
    regardless of whether the caller is in a sync or async client.
    """

    __slots__ = ("_request_id", "_token")

    def __init__(self, request_id: Optional[str]) -> None:
        self._request_id = request_id
        self._token = None

    def __enter__(self) -> "ToolContext":
        self._token = _request_id_ctx.set(self._request_id)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._token is not None:
            _request_id_ctx.reset(self._token)
            self._token = None

    async def __aenter__(self) -> "ToolContext":
        return self.__enter__()

    async def __aexit__(self, *exc: Any) -> None:
        self.__exit__(*exc)


# ---------------------------------------------------------------------------
# Validation + sanitization
# ---------------------------------------------------------------------------

def _byte_len(s: str) -> int:
    return len(s.encode("utf-8"))


def _truncate_string(s: str, max_bytes: int) -> str:
    """Truncates ``s`` to ``max_bytes`` UTF-8 bytes, appending a marker.

    Decodes with ``errors='ignore'`` so a multi-byte sequence cut in the middle
    doesn't produce a replacement character.
    """
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    # Reserve worst-case room for the marker.
    marker_template = " …(truncated {n} bytes)"
    reserve = len(marker_template.format(n=99_999_999).encode("utf-8"))
    truncated_bytes = len(encoded) - (max_bytes - reserve)
    head = encoded[: max_bytes - reserve].decode("utf-8", errors="ignore")
    return head + marker_template.format(n=truncated_bytes)


def validate_tool_name(name: Any) -> str:
    """Validates a tool name; raises if missing/non-string. Truncates if too long."""
    if not isinstance(name, str) or not name:
        raise ValueError("[SignalVault] tool_name must be a non-empty string")
    if _byte_len(name) > MAX_TOOL_NAME_BYTES:
        return _truncate_string(name, MAX_TOOL_NAME_BYTES)
    return name


def sanitize_error(err: Any) -> Optional[str]:
    """Truncates an error message/exception to the server-accepted byte limit."""
    if err is None:
        return None
    s = err if isinstance(err, str) else str(err)
    return _truncate_string(s, MAX_ERROR_BYTES)


def _replace_unserializable(obj: Any, _seen: Optional[set] = None) -> Any:
    """Walks ``obj`` and produces a JSON-safe equivalent.

    - Circular references → ``"[Circular]"``
    - ``bytes`` / ``bytearray`` → utf-8-decoded preview (or repr if undecodable)
    - Numbers, strings, bools, None → unchanged
    - dict/list/tuple → recursive walk
    - Everything else → ``repr(value)`` so the audit still has a useful trace.

    ``set`` of object ids tracks visited containers to detect cycles.
    """
    if _seen is None:
        _seen = set()

    # Primitives and None are already JSON-safe.
    if obj is None or isinstance(obj, (bool, int, float, str)):
        # NaN/Infinity aren't valid JSON; stringify them so json.dumps doesn't
        # produce `NaN` (which is valid Python json output but invalid JSON).
        if isinstance(obj, float):
            if obj != obj:  # NaN
                return "NaN"
            if obj in (float("inf"), float("-inf")):
                return "Infinity" if obj > 0 else "-Infinity"
        return obj

    if isinstance(obj, (bytes, bytearray)):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return f"<bytes: {len(obj)}>"

    if isinstance(obj, dict):
        oid = id(obj)
        if oid in _seen:
            return "[Circular]"
        _seen.add(oid)
        try:
            return {
                # Coerce non-string keys (json requires string keys) so we don't
                # crash on tuple/int keys.
                (k if isinstance(k, str) else str(k)): _replace_unserializable(v, _seen)
                for k, v in obj.items()
            }
        finally:
            _seen.discard(oid)

    if isinstance(obj, (list, tuple, set, frozenset)):
        oid = id(obj)
        if oid in _seen:
            return "[Circular]"
        _seen.add(oid)
        try:
            return [_replace_unserializable(v, _seen) for v in obj]
        finally:
            _seen.discard(oid)

    # Fallback — use repr so e.g. datetime, custom objects, exceptions, etc.
    # produce a deterministic string instead of crashing json.dumps.
    try:
        return repr(obj)
    except Exception:  # pragma: no cover — repr should never raise in practice
        return "<unrepresentable>"


def sanitize_payload(value: Any) -> Any:
    """Returns a JSON-safe version of ``value``, capped at MAX_PAYLOAD_BYTES.

    Never raises — sanitization failures degrade gracefully:

    - Circular references / non-serializable values → marker substitution
    - Oversized values → returns a truncation envelope so the server can tell
      the audit was partial.
    """
    if value is None:
        return None

    safe = _replace_unserializable(value)

    try:
        serialized = json.dumps(safe, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        return {"_signalvault_serialization_error": str(e)}

    if _byte_len(serialized) <= MAX_PAYLOAD_BYTES:
        return safe

    return {
        "_signalvault_truncated": True,
        "_signalvault_original_bytes": _byte_len(serialized),
        "_signalvault_max_bytes": MAX_PAYLOAD_BYTES,
        "preview": _truncate_string(serialized, MAX_PAYLOAD_BYTES - 200),
    }


# ---------------------------------------------------------------------------
# Tool-call options & event shape
# ---------------------------------------------------------------------------

@dataclass
class ToolRecordOptions:
    """Manual tool-call recording options. Mirrors Node ``ToolRecordOptions``."""

    tool_name: str
    tool_input: Any = None
    tool_output: Any = None
    duration_ms: int = 0
    error: Optional[str] = None
    started_at: Optional[str] = None  # ISO8601
    request_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = field(default=None)


def build_tool_call_body(
    *,
    environment: str,
    default_metadata: Dict[str, Any],
    opts: ToolRecordOptions,
) -> Dict[str, Any]:
    """Constructs the JSON body for a POST /v1/events tool_call.

    Validates and sanitizes inputs. Raises ``ValueError`` from
    ``validate_tool_name`` on missing/invalid name (this surfaces from the
    manual API but is caught one level up for the wrapper).
    """
    safe_name = validate_tool_name(opts.tool_name)
    payload: Dict[str, Any] = {
        "tool_name": safe_name,
        "tool_input": sanitize_payload(opts.tool_input),
        "tool_output": sanitize_payload(opts.tool_output),
        "duration_ms": opts.duration_ms,
    }
    safe_error = sanitize_error(opts.error)
    if safe_error is not None:
        payload["error"] = safe_error
    if opts.started_at:
        payload["started_at"] = opts.started_at

    metadata = {**default_metadata, **(opts.metadata or {})}
    body: Dict[str, Any] = {
        "type": "agent.tool_call",
        "environment": environment,
        "metadata": metadata,
        "payload": payload,
    }
    if opts.request_id:
        body["request_id"] = opts.request_id
    return body
