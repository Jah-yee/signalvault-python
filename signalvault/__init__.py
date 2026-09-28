"""SignalVault Python SDK — AI audit logs and guardrails for OpenAI and Anthropic applications."""

# Defined before the imports below: client.py reads it for the User-Agent header.
__version__ = "0.5.0"

from .client import (  # noqa: E402
    AnthropicSignalVaultClient,
    AsyncAnthropicSignalVaultClient,
    AsyncSignalVaultClient,
    SignalVaultBlockedError,
    SignalVaultClient,
    SignalVaultUnavailableError,
    SignalVaultWarning,
)
from .tools import ToolContext, ToolRecordOptions  # noqa: E402

__all__ = [
    "SignalVaultClient",
    "AsyncSignalVaultClient",
    "AnthropicSignalVaultClient",
    "AsyncAnthropicSignalVaultClient",
    "SignalVaultBlockedError",
    "SignalVaultUnavailableError",
    "SignalVaultWarning",
    "ToolContext",
    "ToolRecordOptions",
]
