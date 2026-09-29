"""OpenAI-compatible judge client (FR-002).

Covers frontier OpenAI endpoints and open-weight serverless / local endpoints
via ``base_url`` injection (ADR-006 Update: open-weight serverless is the
default tier, and vLLM exposes the same API — provider + endpoint are env
metadata, the serving stack is part of the environment).

Concurrency is bounded by a semaphore; 429/5xx responses are retried with
exponential backoff + full jitter (:mod:`mdi.providers.base`). The API key is
read from the environment variable named by the config — never from the config
file itself (AGENTS.md §3.4).
"""

import asyncio
import random
from typing import Any, Dict, Optional

import httpx

from mdi.providers.base import (
    DEFAULT_BASE_DELAY_S,
    DEFAULT_MAX_ATTEMPTS,
    RETRY_STATUS,
    JudgeClient,
    JudgeRequest,
    JudgeResponse,
    ProviderError,
    RetryableStatus,
    require_api_key,
    with_retries,
)

DEFAULT_TIMEOUT_S: float = 120.0


class OpenAICompatClient(JudgeClient):
    """Judge client for any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        provider: str,
        max_concurrency: int = 8,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        base_delay_s: float = DEFAULT_BASE_DELAY_S,
        client: Optional[httpx.AsyncClient] = None,
        rng: Optional[random.Random] = None,
        max_tokens_param: str = "max_tokens",
    ) -> None:
        """Configure the endpoint, credentials source and concurrency bound.

        *max_tokens_param* selects the output-cap field name. Chat-completions
        has used ``max_tokens`` historically, but OpenAI's reasoning-capable
        models reject it and require ``max_completion_tokens``; the name is a
        config field rather than a guess from the model string, so a new model
        never silently sends the wrong one.
        """
        self._max_tokens_param = max_tokens_param
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._provider = provider
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._rng = rng
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient(timeout=timeout_s)

    @property
    def provider_name(self) -> str:
        """Provider identifier, recorded as env metadata (ADR-006 Update)."""
        return self._provider

    @property
    def endpoint(self) -> str:
        """Chat-completions endpoint, recorded as env metadata (ADR-006 Update)."""
        return f"{self._base_url}/chat/completions"

    def _headers(self) -> Dict[str, str]:
        """Authorization headers built from the environment (never from config)."""
        return {
            "Authorization": f"Bearer {require_api_key(self._api_key_env)}",
            "Content-Type": "application/json",
        }

    def _body(self, request: JudgeRequest) -> Dict[str, Any]:
        """Build the chat-completions request body for *request*."""
        body: Dict[str, Any] = {
            "model": request.model or self._model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            self._max_tokens_param: request.max_output_tokens,
        }
        if request.seed is not None:
            body["seed"] = request.seed
        return body

    async def _post_once(self, request: JudgeRequest) -> JudgeResponse:
        """Issue one HTTP call, mapping retryable statuses to :class:`RetryableStatus`."""
        response = await self._client.post(
            self.endpoint, headers=self._headers(), json=self._body(request)
        )
        if response.status_code in RETRY_STATUS:
            raise RetryableStatus(response.status_code, response.text)
        if response.status_code >= 400:
            raise ProviderError(f"provider status {response.status_code}: {response.text[:200]}")
        return self._parse(response.json())

    def _parse(self, payload: Dict[str, Any]) -> JudgeResponse:
        """Map an OpenAI-compatible payload onto :class:`JudgeResponse`."""
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"malformed provider payload: {exc}") from exc
        usage = payload.get("usage") or {}
        return JudgeResponse(
            text=text if isinstance(text, str) else "",
            in_tokens=int(usage.get("prompt_tokens", 0)),
            out_tokens=int(usage.get("completion_tokens", 0)),
            model_version=str(payload.get("model", self._model)),
            provider=self._provider,
            endpoint=self.endpoint,
        )

    async def complete(self, request: JudgeRequest) -> JudgeResponse:
        """Send one rendered judge request under the concurrency bound, with backoff."""
        async with self._semaphore:
            return await with_retries(
                lambda: self._post_once(request),
                max_attempts=self._max_attempts,
                base_delay=self._base_delay_s,
                retry_on=(RetryableStatus, httpx.TransportError),
                rng=self._rng,
            )

    async def aclose(self) -> None:
        """Close the underlying HTTP client if this instance owns it."""
        if self._owns_client:
            await self._client.aclose()
