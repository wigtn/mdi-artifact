"""Judge API provider clients (FR-002).

Per ADR-006 (Update): ``provider`` / ``endpoint`` / published precision are env
metadata — the serving stack is part of the environment. ADR-006 Update also
makes open-weight serverless (an OpenAI-compatible endpoint reached by
``base_url`` injection) the default tier.

:func:`build_client` is the single construction point the runner uses, so a
judge is always instantiated from a declared config block and its credentials
always come from the environment (AGENTS.md §3.4).
"""

from typing import Any, Dict

from mdi.providers.anthropic import AnthropicClient
from mdi.providers.base import (
    JudgeClient,
    JudgeRequest,
    JudgeResponse,
    MissingCredentials,
    ProviderError,
)
from mdi.providers.openai_compat import OpenAICompatClient

__all__ = [
    "AnthropicClient",
    "JudgeClient",
    "JudgeRequest",
    "JudgeResponse",
    "MissingCredentials",
    "OpenAICompatClient",
    "ProviderError",
    "build_client",
]


def build_client(judge: Dict[str, Any], max_concurrency: int) -> JudgeClient:
    """Construct the judge client declared by a config ``judge:`` block.

    ``provider_kind`` selects the wire protocol (``openai_compat`` or
    ``anthropic``); ``api_key_env`` names the environment variable holding the
    credential — the value itself never appears in configs.
    """
    kind = str(judge.get("provider_kind", "openai_compat"))
    model = str(judge["model"])
    provider = str(judge.get("provider", kind))
    if kind == "openai_compat":
        return OpenAICompatClient(
            model=model,
            base_url=str(judge["endpoint"]),
            api_key_env=str(judge["api_key_env"]),
            provider=provider,
            max_concurrency=max_concurrency,
            max_tokens_param=str(judge.get("max_tokens_param", "max_tokens")),
        )
    if kind == "anthropic":
        return AnthropicClient(
            model=model,
            api_key_env=str(judge.get("api_key_env", "ANTHROPIC_API_KEY")),
            base_url=str(judge.get("endpoint", "https://api.anthropic.com/v1")),
            provider=provider,
            max_concurrency=max_concurrency,
        )
    raise ValueError(f"unknown judge.provider_kind: {kind!r}")
