"""Anthropic Messages-API judge client (FR-002).

Implemented directly over ``httpx`` rather than the SDK: one endpoint, one
request shape, and no extra dependency to pin for reproducibility. Model
snapshot/version is taken from the response's ``model`` field and recorded per
record (``judge_model_version``, PRD §5.2); provider and endpoint are env
metadata (ADR-006 Update).

Concurrency is bounded by a semaphore; 429/5xx responses are retried with
exponential backoff + full jitter. The API key is read from the environment
variable named by the config — never from the config file (AGENTS.md §3.4).
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

DEFAULT_BASE_URL: str = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION: str = "2023-06-01"
DEFAULT_TIMEOUT_S: float = 120.0


class AnthropicClient(JudgeClient):
    """Judge client for the Anthropic Messages API."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: str = DEFAULT_BASE_URL,
        provider: str = "anthropic",
        max_concurrency: int = 4,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        base_delay_s: float = DEFAULT_BASE_DELAY_S,
        client: Optional[httpx.AsyncClient] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Configure the endpoint, credentials source and concurrency bound."""
        self._model = model
        self._api_key_env = api_key_env
        self._base_url = base_url.rstrip("/")
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
        """Messages endpoint, recorded as env metadata (ADR-006 Update)."""
        return f"{self._base_url}/messages"

    def _headers(self) -> Dict[str, str]:
        """Auth + version headers built from the environment (never from config)."""
        return {
            "x-api-key": require_api_key(self._api_key_env),
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    def _body(self, request: JudgeRequest) -> Dict[str, Any]:
        """Build the Messages request body for *request*.

        The Anthropic API exposes no ``seed`` parameter; the runner still
        records the per-repeat seed as env metadata (ADR-011 D1) so the intended
        sampling axis stays auditable.
        """
        return {
            "model": request.model or self._model,
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}],
        }

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
        """Map a Messages payload onto :class:`JudgeResponse`."""
        blocks = payload.get("content")
        if not isinstance(blocks, list):
            raise ProviderError("malformed provider payload: content is not a list")
        text = "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = payload.get("usage") or {}
        return JudgeResponse(
            text=text,
            in_tokens=int(usage.get("input_tokens", 0)),
            out_tokens=int(usage.get("output_tokens", 0)),
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
