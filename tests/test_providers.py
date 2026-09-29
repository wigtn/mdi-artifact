"""Tests for the async judge clients: credentials, retries, usage capture, provenance.

No test touches the network: every client is constructed with an
``httpx.MockTransport`` and backoff delays are set to zero.
"""

import asyncio
import json
import random
from typing import Any, Callable, Dict, List

import httpx
import pytest

from mdi.providers import build_client
from mdi.providers.anthropic import AnthropicClient
from mdi.providers.base import (
    JudgeRequest,
    MissingCredentials,
    ProviderError,
    RetryableStatus,
    backoff_delay,
    require_api_key,
    with_retries,
)
from mdi.providers.openai_compat import OpenAICompatClient

KEY_VAR = "MDI_TEST_PROVIDER_KEY"


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    """An httpx client whose transport is the given in-process handler."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def openai_payload(content: str = '{"score": 4}') -> Dict[str, Any]:
    """A minimal OpenAI-compatible chat-completions response body."""
    return {
        "model": "test/judge-1-2026-07-31",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 812, "completion_tokens": 41},
    }


def anthropic_payload(content: str = '{"score": 4}') -> Dict[str, Any]:
    """A minimal Anthropic Messages response body."""
    return {
        "model": "claude-test-20260731",
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": 812, "output_tokens": 41},
    }


def request() -> JudgeRequest:
    """A rendered judge request used across the provider tests."""
    return JudgeRequest(
        model="test/judge-1", prompt="Rate this.", temperature=1.0, max_output_tokens=32, seed=7
    )


# --------------------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------------------


def test_require_api_key_raises_when_the_env_var_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing key fails fast, naming the variable and never a value (AGENTS.md §3.4)."""
    # Given: no key in the environment
    monkeypatch.delenv(KEY_VAR, raising=False)
    # When / Then: the error names the variable
    with pytest.raises(MissingCredentials, match=KEY_VAR):
        require_api_key(KEY_VAR)


def test_require_api_key_returns_the_value_when_the_env_var_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The key is read from os.environ only."""
    # Given: a key in the environment
    monkeypatch.setenv(KEY_VAR, "secret-value")
    # When / Then
    assert require_api_key(KEY_VAR) == "secret-value"


def test_openai_client_raises_before_any_http_call_when_the_key_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Credential checks happen before the request is sent, so nothing leaves the process."""
    # Given: an unset key and a transport that would fail the test if used
    monkeypatch.delenv(KEY_VAR, raising=False)
    seen: List[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        """Record any request that reaches the transport."""
        seen.append(req)
        return httpx.Response(200, json=openai_payload())

    client = OpenAICompatClient(
        model="test/judge-1",
        base_url="https://example.invalid/v1",
        api_key_env=KEY_VAR,
        provider="test",
        client=mock_client(handler),
    )
    # When / Then: the call fails and no request was issued
    with pytest.raises(ProviderError, match=KEY_VAR):
        asyncio.run(client.complete(request()))
    assert seen == []


# --------------------------------------------------------------------------------------
# OpenAI-compatible client
# --------------------------------------------------------------------------------------


def test_openai_client_captures_usage_and_model_version_when_the_call_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-call usage and the provider's model echo become record provenance (PRD §5.2)."""
    # Given: a successful chat-completions response
    monkeypatch.setenv(KEY_VAR, "secret-value")
    captured: Dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        """Capture the outgoing request and answer with a fixed payload."""
        captured["url"] = str(req.url)
        captured["auth"] = req.headers.get("authorization")
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=openai_payload())

    client = OpenAICompatClient(
        model="test/judge-1",
        base_url="https://serverless.example/v1",
        api_key_env=KEY_VAR,
        provider="serverless",
        client=mock_client(handler),
    )
    # When: one request is completed
    response = asyncio.run(client.complete(request()))
    # Then: usage, version and env metadata come back; base_url injection is honored
    assert response.text == '{"score": 4}'
    assert (response.in_tokens, response.out_tokens) == (812, 41)
    assert response.model_version == "test/judge-1-2026-07-31"
    assert response.provider == "serverless"
    assert response.endpoint == "https://serverless.example/v1/chat/completions"
    assert captured["url"] == "https://serverless.example/v1/chat/completions"
    assert captured["auth"] == "Bearer secret-value"
    assert captured["body"]["temperature"] == 1.0
    assert captured["body"]["seed"] == 7


def test_openai_client_retries_and_succeeds_when_the_provider_returns_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """429 is retried with backoff rather than surfaced as a failure."""
    # Given: a provider that rate-limits twice, then succeeds
    monkeypatch.setenv(KEY_VAR, "secret-value")
    attempts: List[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        """Return 429 for the first two attempts, then 200."""
        attempts.append(1)
        if len(attempts) <= 2:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=openai_payload())

    client = OpenAICompatClient(
        model="test/judge-1",
        base_url="https://serverless.example/v1",
        api_key_env=KEY_VAR,
        provider="serverless",
        client=mock_client(handler),
        base_delay_s=0.0,
    )
    # When: the call is made
    response = asyncio.run(client.complete(request()))
    # Then: three attempts and a parsed response
    assert len(attempts) == 3
    assert response.in_tokens == 812


def test_openai_client_raises_provider_error_when_backoff_is_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permanently unavailable provider raises ProviderError (PRD §5.1)."""
    # Given: a provider stuck on 503
    monkeypatch.setenv(KEY_VAR, "secret-value")
    attempts: List[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        """Always fail with a retryable status."""
        attempts.append(1)
        return httpx.Response(503, text="unavailable")

    client = OpenAICompatClient(
        model="test/judge-1",
        base_url="https://serverless.example/v1",
        api_key_env=KEY_VAR,
        provider="serverless",
        client=mock_client(handler),
        base_delay_s=0.0,
        max_attempts=3,
    )
    # When / Then
    with pytest.raises(ProviderError, match="failed after 3 attempts"):
        asyncio.run(client.complete(request()))
    assert len(attempts) == 3


def test_openai_client_does_not_retry_when_the_status_is_not_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 400 is a bug in our request, not transient — retrying would just burn budget."""
    # Given: a provider rejecting the request
    monkeypatch.setenv(KEY_VAR, "secret-value")
    attempts: List[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        """Always answer 400."""
        attempts.append(1)
        return httpx.Response(400, text="bad request")

    client = OpenAICompatClient(
        model="test/judge-1",
        base_url="https://serverless.example/v1",
        api_key_env=KEY_VAR,
        provider="serverless",
        client=mock_client(handler),
        base_delay_s=0.0,
    )
    # When / Then: one attempt only
    with pytest.raises(ProviderError, match="provider status 400"):
        asyncio.run(client.complete(request()))
    assert len(attempts) == 1


# --------------------------------------------------------------------------------------
# Anthropic client
# --------------------------------------------------------------------------------------


def test_anthropic_client_parses_content_blocks_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Messages payload maps onto the same JudgeResponse shape."""
    # Given: a Messages response
    monkeypatch.setenv(KEY_VAR, "secret-value")
    captured: Dict[str, Any] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        """Capture headers and answer with a fixed Messages payload."""
        captured["url"] = str(req.url)
        captured["key"] = req.headers.get("x-api-key")
        captured["version"] = req.headers.get("anthropic-version")
        return httpx.Response(200, json=anthropic_payload())

    client = AnthropicClient(model="claude-test", api_key_env=KEY_VAR, client=mock_client(handler))
    # When
    response = asyncio.run(client.complete(request()))
    # Then
    assert response.text == '{"score": 4}'
    assert (response.in_tokens, response.out_tokens) == (812, 41)
    assert response.model_version == "claude-test-20260731"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["key"] == "secret-value"
    assert captured["version"] == "2023-06-01"


# --------------------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------------------


def test_backoff_delay_grows_exponentially_and_stays_within_the_ceiling() -> None:
    """Full-jitter backoff never exceeds min(max_delay, base * 2^(n-1))."""
    # Given: a seeded RNG
    rng = random.Random(0)
    # When / Then: every draw is inside its attempt's ceiling
    for attempt in range(1, 8):
        ceiling = min(30.0, 1.0 * (2 ** (attempt - 1)))
        assert 0.0 <= backoff_delay(attempt, 1.0, 30.0, rng) <= ceiling


def test_with_retries_sleeps_between_attempts_and_reraises_as_provider_error() -> None:
    """The retry helper backs off between attempts and converts exhaustion to ProviderError."""
    # Given: an operation that always raises a retryable status
    slept: List[float] = []

    async def sleeper(seconds: float) -> None:
        """Record the requested delay instead of waiting."""
        slept.append(seconds)

    async def always_429() -> str:
        """Fail with a retryable status."""
        raise RetryableStatus(429, "slow down")

    async def scenario() -> None:
        """Drive the retry helper to exhaustion."""
        await with_retries(
            always_429, max_attempts=4, base_delay=1.0, sleep=sleeper, rng=random.Random(0)
        )

    # When / Then: three sleeps for four attempts
    with pytest.raises(ProviderError, match="failed after 4 attempts"):
        asyncio.run(scenario())
    assert len(slept) == 3


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


def test_build_client_selects_the_wire_protocol_declared_in_the_config() -> None:
    """The runner constructs judges only from a declared config block."""
    # Given: two judge blocks
    openai_judge: Dict[str, Any] = {
        "model": "m",
        "provider_kind": "openai_compat",
        "endpoint": "https://x.example/v1",
        "api_key_env": KEY_VAR,
        "provider": "x",
    }
    anthropic_judge: Dict[str, Any] = {"model": "m", "provider_kind": "anthropic"}
    # When / Then
    assert isinstance(build_client(openai_judge, 2), OpenAICompatClient)
    assert isinstance(build_client(anthropic_judge, 2), AnthropicClient)
    with pytest.raises(ValueError, match="unknown judge.provider_kind"):
        build_client({"model": "m", "provider_kind": "carrier-pigeon"}, 2)
