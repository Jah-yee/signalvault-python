"""SignalVault Python SDK — AI audit logs and guardrails for OpenAI and Anthropic applications."""

from .client import (
    AnthropicSignalVaultClient,
    AsyncAnthropicSignalVaultClient,
    AsyncSignalVaultClient,
    SignalVaultClient,
)
from .tools import ToolContext, ToolRecordOptions

__all__ = [
    "SignalVaultClient",
    "AsyncSignalVaultClient",
    "AnthropicSignalVaultClient",
    "AsyncAnthropicSignalVaultClient",
    "ToolContext",
    "ToolRecordOptions",
]
__version__ = "0.4.0"
