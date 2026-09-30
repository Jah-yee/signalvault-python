# signalvault

AI audit logs and guardrails for your OpenAI and Anthropic Python applications.

## Installation

```bash
# OpenAI only
pip install signalvault openai

# Anthropic only
pip install signalvault[anthropic]

# Both
pip install signalvault openai signalvault[anthropic]
```

## Quick Start — OpenAI (sync)

```python
import os
from signalvault import SignalVaultClient

client = SignalVaultClient(
    api_key="sk_live_your_signalvault_key",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.signalvault.io",
    environment="production",
)

# Use exactly like OpenAI SDK
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Quick Start — OpenAI (async, FastAPI / async Django)

```python
import os
from signalvault import AsyncSignalVaultClient

client = AsyncSignalVaultClient(
    api_key="sk_live_your_signalvault_key",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://api.signalvault.io",
)

response = await client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Quick Start — Anthropic

```python
import os
from signalvault import AnthropicSignalVaultClient

client = AnthropicSignalVaultClient(
    api_key="sk_live_your_signalvault_key",
    anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
    base_url="https://api.signalvault.io",
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=1024,
)
print(response.content[0].text)
```

## Streaming

Streaming is fully supported for all clients and providers:

```python
# OpenAI streaming (sync)
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

# OpenAI streaming (async)
stream = await async_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True,
)
async for chunk in stream:
    if chunk.choices and chunk.choices[0].delta:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

# Anthropic streaming
stream = anthropic_client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Write a poem"}],
    max_tokens=1024,
    stream=True,
)
for event in stream:
    if event.type == "content_block_delta":
        print(event.delta.text or "", end="", flush=True)
```

## Agent Tool-Use Capture

Log every tool invocation made by your agents — name, input, output, duration, and any error — as auditable events alongside your LLM calls.

### Wrapper API (recommended)

```python
@client.tool("fetch_weather")
def fetch_weather(city: str):
    res = httpx.get(f"https://api.example.com/weather?city={city}")
    return res.json()

# Call it like the original — SignalVault auto-times and audits.
weather = fetch_weather("London")
```

The wrapper records the call asynchronously (no impact on your tool's latency) and passes the result through unchanged. Errors are recorded and re-raised.

```python
# Async client + async tool
@async_client.tool("fetch_weather")
async def fetch_weather(city: str):
    async with httpx.AsyncClient() as http:
        res = await http.get(f"https://api.example.com/weather?city={city}")
        return res.json()

weather = await fetch_weather("London")
```

### Manual API

```python
# Sync — blocks on the HTTP send so errors surface to the caller
client.tools.record(
    tool_name="fetch_weather",
    tool_input={"city": "London"},
    tool_output={"temp": 12.3},
    duration_ms=142,
)

# Async
await async_client.tools.record(
    tool_name="fetch_weather",
    tool_input={"city": "London"},
    tool_output={"temp": 12.3},
    duration_ms=142,
)
```

### Linking tool calls to a parent LLM turn

Wrap your agent loop in `with_context` and tool calls inside auto-correlate to the given `request_id`:

```python
# Sync
with client.with_context(request_id="agent-turn-abc"):
    llm_response = client.chat.completions.create(...)
    fetch_weather("London")  # auto-linked to 'agent-turn-abc'

# Async
async with async_client.with_context(request_id="agent-turn-abc"):
    llm_response = await async_client.chat.completions.create(...)
    await fetch_weather("London")  # auto-linked to 'agent-turn-abc'
```

Without `with_context`, tool calls are recorded as orphans (no parent request).

### What gets captured (and what to keep out)

When you wrap a tool or call `tools.record()`, SignalVault captures:

- `tool_name` (truncated to 200 bytes)
- `tool_input` — the function arguments, JSON-serialized (capped at 256 KB; oversize values are truncated with a marker)
- `tool_output` — the function return value, JSON-serialized (same 256 KB cap)
- `error` if the tool raises (truncated to 1900 bytes)
- `duration_ms`, `started_at`, and any `metadata` you attach

`tool_input` and `tool_output` are encrypted at rest server-side. `tool_name`, `error`, `started_at` and `metadata` are stored as sent, **not encrypted**. All of it goes on the wire to SignalVault's API: **if you pass user PII, secrets, or API keys as tool arguments, those values will leave your process and be stored in SignalVault.** Recommendations:

- Sanitize sensitive arguments before invoking the wrapped tool, or use the manual `tools.record()` API and pass a redacted copy.
- Don't put secrets in error messages — they end up in `error` verbatim and unencrypted.
- Use `metadata` for non-sensitive identifiers (`user_id`, `feature`, `workspace_id`); avoid putting raw user content in metadata.

## Metadata

Attach contextual metadata to every event for user attribution, analytics, and audit trails:

```python
# Set defaults at client level
client = SignalVaultClient(
    api_key="sk_live_...",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    metadata={"workspace_id": "ws_abc", "env": "production"},
)

# Override per-call
response = client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    sv_metadata={"user_id": "u_123", "feature": "support-chat"},
)
```

`sv_metadata` goes to SignalVault. A plain `metadata=` argument is passed through to the provider untouched (for example Anthropic's `metadata={"user_id": ...}`).

## When SignalVault Is Unavailable

The pre-flight guardrail check is in your request's critical path. If it cannot return a decision — timeout, network error, invalid or revoked API key (401), inactive subscription (402), access denied (403), rate or trial limit (429), server error, or an invalid response — `fail_mode` decides what happens:

- `"open"` (default): the request goes to the provider **without** guardrails, and the SDK emits a `SignalVaultWarning` (at most once a minute per cause, whether or not `debug` is on). If SignalVault was unreachable, timed out or errored, the request is still recorded in the background, marked `preflight_unavailable`, so it appears in your audit log.
- `"closed"`: the SDK raises `SignalVaultUnavailableError` and the provider is never called. Use this if every request must be checked.

`fail_mode` applies to the pre-flight check, so it has no effect in mirror mode.

**Rate limit.** The ingest API allows 120 events per minute per app, and each call sends two (request and response), plus one per tool call. Above roughly 60 LLM calls a minute, pre-flight checks are rate limited: with `fail_mode="open"` those requests go through unchecked. If you run at that volume and need enforcement, use `fail_mode="closed"`.

```python
client = SignalVaultClient(
    api_key="sk_live_...",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    fail_mode="closed",      # "open" | "closed". Default: "open"
    preflight_timeout=2.0,   # seconds — pre-flight check. Default: 2.0
    timeout=30.0,            # seconds — background/post-flight calls. Default: 30.0
)
```

`base_url` defaults to `https://api.signalvault.io`. Plain `http://` is refused except for `localhost`, as is a URL with a query string or fragment.

Audit events (responses, mirror-mode events, tool calls) are sent in the background. Each carries an `event_id`, so a retry is never double-counted. The SDK retries once on connection errors and 5xx (not on timeouts). A rate-limited (429) event is retried only if `Retry-After` is 5 seconds or less; the SignalVault API currently asks for 60, so in practice those events are dropped, with a warning. `tools.record()` does not retry.

## Shutting Down

Background events are queued in memory. Wait for them before a short-lived process exits:

```python
client.close()          # sync: sends queued events (up to 5s), then closes the connection pool
await client.aclose()   # async: same; required — pending async events are lost if the loop exits first

# Or only wait, keeping the client open:
client.flush()
await async_client.flush()
```

## Error Handling

```python
from signalvault import SignalVaultBlockedError, SignalVaultUnavailableError

try:
    response = client.chat.completions.create(model="gpt-4", messages=[...])
except SignalVaultBlockedError as e:
    print("Blocked by guardrails:", [v.type for v in e.violations])
    print("Details:", e.dashboard_url)
except SignalVaultUnavailableError as e:
    # Only raised with fail_mode="closed"
    print("Guardrail check unavailable:", e.status)
```

Both are subclasses of `RuntimeError`.

## Mirror Mode

In mirror mode, requests go directly to the AI provider first and SignalVault audits them asynchronously — no latency added, never blocks:

```python
client = SignalVaultClient(
    api_key="sk_live_...",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    mirror_mode=True,
)
```

## Features

- **Automatic Logging** — Every request and response is recorded
- **Pre-flight Guardrails** — Block requests that break your rules before they reach the AI provider
- **Redaction at rest** — Rules with the `redact` action remove matches from what SignalVault stores. The request sent to the AI provider is **not** modified.
- **PII Detection** — Detect emails, phone numbers, SSNs
- **Secret Detection** — Block API keys and tokens
- **Token Limits** — Enforce cost controls
- **Model Allowlists** — Restrict which models can be used
- **Streaming** — Full streaming support for OpenAI and Anthropic
- **Async Support** — `AsyncSignalVaultClient` and `AsyncAnthropicSignalVaultClient` for async codebases
- **Mirror Mode** — Observe without blocking
- **Metadata** — Tag every event with user_id, feature, workspace_id, etc.
- **Multi-provider** — OpenAI and Anthropic/Claude support

## License

MIT
