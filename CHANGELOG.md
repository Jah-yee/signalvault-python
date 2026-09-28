# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] — Unreleased

Security and reliability release.

### Breaking changes

- **Default `base_url` is now `https://api.signalvault.io`** (was `http://localhost:4000`). Pass `base_url` explicitly for local development.
- **Plain `http://` base URLs are refused** except for `localhost`, `127.0.0.1` and `::1`, because the API key and prompts would travel unencrypted. A `base_url` with a query string or fragment is refused too.
- **`metadata=` is no longer taken over by SignalVault.** It is passed through to the provider (OpenAI's `metadata`, Anthropic's `metadata={"user_id": ...}`). Use `sv_metadata=` for SignalVault metadata; the `metadata` fallback was deprecated in 0.3.0.
- **Blocked requests raise `SignalVaultBlockedError`** (a `RuntimeError` subclass) with `violations`, `request_id` and `dashboard_url`. The message lists violation types instead of the full violation dicts.
- **`Decision.redactions`** is a list of `{"type", "count"}` dicts, matching the API.

### Security

- A failed guardrail check is no longer silent. An invalid key (401), inactive account (402), denied access (403), rate or trial limit (429), server error, timeout or invalid response now emits a `SignalVaultWarning` (at most once a minute per cause) even with `debug=False`.
- New `fail_mode="open" | "closed"`. With `"closed"`, the SDK raises `SignalVaultUnavailableError` instead of calling the provider unchecked. Default stays `"open"`.
- The Anthropic `system` prompt is now included in the pre-flight scan and the audit record.
- HTTP clients never follow redirects, so the API key is never re-sent to another URL.
- When the pre-flight check could not be completed (unreachable, timeout, 5xx, invalid response) and the request went ahead unchecked, the request is now recorded in the background, marked `preflight_unavailable`. Previously only the response was sent, which the API rejects, so these calls were missing from the audit log. After a refused check (401/402/403/429) no audit events are sent.
- `SignalVaultWarning` is reported at the caller's line and repeats after the one-minute window. Previously Python's default warning filter showed each message only once per process, so a later outage could go unreported.

### Fixes

- A 200 pre-flight response with a non-JSON body, or with wrongly typed fields (for example a non-string `decision` or non-list `violations`), no longer raises into the caller.
- Deeply nested tool inputs no longer raise `RecursionError`; nesting beyond 100 levels is replaced with `"[MaxDepth]"`.
- Tool-call events are built when the tool returns, so later mutation of the arguments no longer changes what is recorded. An invalid tool name no longer breaks the wrapped tool.
- Async clients keep references to background tasks, so audit events are no longer lost to garbage collection. Scheduling without a running loop now warns instead of dropping silently.
- Anthropic responses record every text block, not only the first.

### Added

- `flush()` on sync clients and `await flush()` on async clients; `close()` / `aclose()` now send queued events first.
- Every background event carries an `event_id`. The SDK retries once on connection errors and 5xx (not timeouts, so a slow server holds a worker for at most one `timeout` per event), and on 429 only when `Retry-After` is 5 seconds or less, which the SignalVault API does not currently send. `tools.record()` does not retry; its events also carry an `event_id`.
- `Accept: application/json` and `User-Agent: signalvault-python/<version>` headers.
- LICENSE file.

---

## [0.4.0] — 2026-04-29

### Added

- **Agent tool-use capture.** `client.tool(name, fn)` wraps sync or async functions and records each call as an `agent.tool_call` event; `client.tools.record(...)` records manually; `client.with_context(request_id=...)` links tool calls to a parent request.

---

## [0.3.0] — 2026-03-26

### Added

- **Streaming support** for all four clients (`SignalVaultClient`, `AsyncSignalVaultClient`, `AnthropicSignalVaultClient`, `AsyncAnthropicSignalVaultClient`). Pass `stream=True` to `chat.completions.create()` or `messages.create()` and iterate the returned generator as normal. SignalVault fires the `ai.response` audit event automatically after the stream is fully consumed (or closed early).
- **`sv_metadata` parameter** on every `create()` call. Per-call metadata is merged on top of the client-level `metadata` config. The old `metadata` kwarg still works but now emits a `DeprecationWarning` unconditionally.
- **`warn` decision handling** — when the SignalVault backend returns `decision: "warn"` and the client is configured with `debug=True`, a `UserWarning` is now emitted. Previously this was silently ignored in the async clients.
- **Context manager support** for all clients. Sync clients implement `__enter__`/`__exit__`; async clients implement `__aenter__`/`__aexit__`. Use `with SignalVaultClient(...) as client:` or `async with AsyncSignalVaultClient(...) as client:` to ensure the HTTP connection pool is properly closed.
- **`close()` / `aclose()` methods** on all clients for explicit resource cleanup without a context manager.

### Changed

- **Anthropic streaming** now uses `anthropic.messages.stream()` as a proper context manager instead of the undocumented raw iterator path. This is more robust across Anthropic SDK versions. The `stream=True` kwarg is no longer forwarded to the underlying Anthropic client.
- **Anthropic stream text extraction** now checks `event.delta.type == "text_delta"` before appending content, matching the Anthropic SDK's documented event shape and aligning with the Node SDK behaviour.
- **`DeprecationWarning` for `metadata` kwarg** now fires regardless of `debug` mode. Deprecation warnings are always relevant to callers.
- **Internal refactor** — `_BaseSyncClient` and `_BaseAsyncClient` base classes now hold all shared HTTP, config, and audit logic. The four public client classes inherit from these bases, eliminating ~40% code duplication and ensuring fixes are applied consistently.
- **`_parse_decision`** now filters unknown fields from violation objects before constructing `Violation` dataclasses. This prevents `TypeError` crashes if the SignalVault backend adds new violation fields in a future release.
- **`asyncio.create_task()`** calls in async stream `finally` blocks are now guarded with `try/except RuntimeError` to handle the case where the event loop has already shut down, preventing a confusing error from masking the original exception.
- **`_wrap_stream` return type annotations** corrected to `Generator[Any, None, None]` and `AsyncGenerator[Any, None]`.

### Fixed

- `__version__` in `signalvault/__init__.py` was still `"0.2.0"` while `pyproject.toml` declared `0.3.0`. Both now report `0.3.0`.
- Async `_normal` methods in `_AsyncChatCompletions` and `_AsyncAnthropicMessages` were missing the `warn` decision branch, silently swallowing guardrail warnings.
- `httpx.Client` and `httpx.AsyncClient` instances were never closed, leaking OS connections and triggering `ResourceWarning` on garbage collection.

---

## [0.2.0]

Initial public release with sync and async OpenAI and Anthropic client wrappers, preflight guardrails, mirror mode, and configurable timeouts.
