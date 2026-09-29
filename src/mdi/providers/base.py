"""Abstract async judge client interface + shared retry policy (FR-002).

Concrete clients (frontier APIs, OpenAI-compatible serverless/local endpoints)
implement this interface. Provider identity, endpoint and the model version the
provider echoes back are recorded as env metadata / record provenance
(ADR-006 Update, PRD §5.2). Clients are invoked only by the runner behind the
cost gate (AGENTS.md §3.2).

Credentials are read from ``os.environ`` only — never from config files, never
logged (AGENTS.md §3.4). A missing key fails fast with the variable name.
"""

import abc
import asyncio
import os
import random
from typing import Awaitable, Callable, FrozenSet, Optional, Tuple, Type, TypeVar

DEFAULT_MAX_ATTEMPTS: int = 5
DEFAULT_BASE_DELAY_S: float = 1.0
DEFAULT_MAX_DELAY_S: float = 30.0
RETRY_STATUS: FrozenSet[int] = frozenset({408, 409, 429, 500, 502, 503, 504, 529})

T = TypeVar("T")


class ProviderError(Exception):
    """Provider call failed after backoff was exhausted (PRD §5.1)."""


class MissingCredentials(ProviderError):
    """The provider's API-key environment variable is unset (AGENTS.md §3.4)."""


class RetryableStatus(ProviderError):
    """A provider response whose status is worth retrying (429 / 5xx)."""

    def __init__(self, status_code: int, body: str) -> None:
        """Record the HTTP status and a truncated body for diagnostics."""
        super().__init__(f"retryable provider status {status_code}: {body[:200]}")
        self.status_code = status_code


class JudgeRequest:
    """One rendered judge call (the prompt is already fully materialized)."""

    def __init__(
        self,
        model: str,
        prompt: str,
        temperature: float,
        max_output_tokens: int,
        seed: Optional[int] = None,
    ) -> None:
        """Store the request fields; no rendering or templating happens here."""
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.seed = seed


class JudgeResponse:
    """One judge response plus the usage/provenance the score record needs."""

    def __init__(
        self,
        text: str,
        in_tokens: int,
        out_tokens: int,
        model_version: str,
        provider: str,
        endpoint: str,
    ) -> None:
        """Store the verbatim response text and its usage/provenance metadata."""
        self.text = text
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.model_version = model_version
        self.provider = provider
        self.endpoint = endpoint


class JudgeClient(abc.ABC):
    """Abstract async judge API client with bounded concurrency."""

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Provider identifier, recorded as env metadata (ADR-006 Update)."""

    @property
    @abc.abstractmethod
    def endpoint(self) -> str:
        """Endpoint URL, recorded as env metadata (ADR-006 Update)."""

    @abc.abstractmethod
    async def complete(self, request: JudgeRequest) -> JudgeResponse:
        """Send one rendered judge request and return the parsed provider response."""

    async def aclose(self) -> None:
        """Release provider resources (no-op by default)."""
        return None


def require_api_key(env_var: str) -> str:
    """Read an API key from the environment or fail with the variable name.

    The value is never echoed — only the variable name appears in the error.
    """
    key = os.environ.get(env_var, "").strip()
    if not key:
        raise MissingCredentials(
            f"environment variable {env_var} is unset or empty; "
            "put judge API keys in .env or the environment only (AGENTS.md §3.4)"
        )
    return key


def backoff_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY_S,
    max_delay: float = DEFAULT_MAX_DELAY_S,
    rng: Optional[random.Random] = None,
) -> float:
    """Exponential backoff with full jitter for a 1-based *attempt* number."""
    ceiling = min(max_delay, base_delay * (2 ** (attempt - 1)))
    source = rng if rng is not None else random
    return source.uniform(0.0, ceiling)


async def with_retries(
    operation: Callable[[], Awaitable[T]],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_S,
    max_delay: float = DEFAULT_MAX_DELAY_S,
    retry_on: Tuple[Type[BaseException], ...] = (RetryableStatus,),
    rng: Optional[random.Random] = None,
    sleep: Optional[Callable[[float], Awaitable[None]]] = None,
) -> T:
    """Run *operation*, retrying retryable failures with exponential backoff + jitter.

    Raises :class:`ProviderError` once the attempt budget is exhausted (PRD §5.1).
    *sleep* is injectable so tests never wait on the wall clock.
    """
    sleeper = sleep if sleep is not None else asyncio.sleep
    last: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await operation()
        except retry_on as exc:
            last = exc
            if attempt == max_attempts:
                break
            await sleeper(backoff_delay(attempt, base_delay, max_delay, rng))
    raise ProviderError(f"provider call failed after {max_attempts} attempts: {last}") from last
