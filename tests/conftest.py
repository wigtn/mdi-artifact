"""Shared fixtures: an offline config + a mock judge client (no network in tests)."""

import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import pytest

from mdi.providers.base import JudgeClient, JudgeRequest, JudgeResponse

PROMPT_TEMPLATE = (
    "Rate the summary 1-5.\n\nSource:\n{{source}}\n\nSummary:\n{{candidate}}\n\n"
    'Reply with {"score": <1-5>}.\n'
)

MODEL = "test/judge-1"


class MockJudgeClient(JudgeClient):
    """Deterministic judge stub that counts calls — the money-safety test double."""

    def __init__(
        self,
        responses: Optional[Sequence[str]] = None,
        default_response: str = '{"score": 4}',
        in_tokens: int = 100,
        out_tokens: int = 10,
    ) -> None:
        """Configure a scripted response sequence and the usage each call reports."""
        self.responses = list(responses) if responses is not None else []
        self.default_response = default_response
        self.in_tokens = in_tokens
        self.out_tokens = out_tokens
        self.calls: List[JudgeRequest] = []

    @property
    def provider_name(self) -> str:
        """Provider identifier recorded as env metadata."""
        return "mock"

    @property
    def endpoint(self) -> str:
        """Endpoint recorded as env metadata."""
        return "mock://judge"

    async def complete(self, request: JudgeRequest) -> JudgeResponse:
        """Return the next scripted response and record the request."""
        index = len(self.calls)
        self.calls.append(request)
        text = self.responses[index] if index < len(self.responses) else self.default_response
        return JudgeResponse(
            text=text,
            in_tokens=self.in_tokens,
            out_tokens=self.out_tokens,
            model_version=f"{MODEL}-2026-07-31",
            provider=self.provider_name,
            endpoint=self.endpoint,
        )


def write_items_jsonl(path: str, n_items: int = 3, systems: Sequence[str] = ("s_A", "s_B")) -> str:
    """Write a tiny local item file in the loader's JSONL schema."""
    with open(path, "w", encoding="utf-8") as fh:
        for index in range(n_items):
            record = {
                "item_id": f"i_{index:03d}",
                "source": f"Source document {index}. " * 4,
                "outputs": {sid: f"Summary {sid} for item {index}." for sid in systems},
            }
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def base_config(items_path: str) -> Dict[str, Any]:
    """A minimal, fully offline experiment config (jsonl item source)."""
    return {
        "experiment": "test_exp",
        "benchmark": "test_bench",
        "judge": {
            "model": MODEL,
            "model_version": "test-version",
            "tier": "api",
            "provider": "mock",
            "provider_kind": "openai_compat",
            "endpoint": "mock://judge",
            "api_key_env": "MDI_TEST_KEY",
            "serving_engine": None,
            "quantization": None,
        },
        "prompt": {
            "prompt_id": "p_test",
            "paraphrase_id": "pp_0",
            "templates": {"likert5": PROMPT_TEMPLATE},
        },
        "grid": {"task": ["summarization"], "scale": ["likert5"], "temperature": [1.0]},
        "inputs": {"kind": "jsonl", "path": items_path},
        "repeats": {"n": 4, "max_attempts_factor": 2, "seed_base": 1000},
        "pilot": {"n_items": 2, "n_repeats": 2},
        "budget": {"pilot_cap_usd": 5.0, "cumulative_cap_usd": 10.0},
        "pricing": {MODEL: {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 2.0}},
        "runtime": {"concurrency": 4, "max_output_tokens": 32, "chars_per_token": 4.0},
    }


@pytest.fixture()
def items_path(tmp_path: Any) -> str:
    """Path to a freshly written local item file."""
    return write_items_jsonl(os.path.join(str(tmp_path), "items.jsonl"))


@pytest.fixture()
def config(items_path: str) -> Dict[str, Any]:
    """A fresh deep copy of the offline base config."""
    return copy.deepcopy(base_config(items_path))


@pytest.fixture()
def data_dir(tmp_path: Any) -> str:
    """An isolated data tree root for one test."""
    return os.path.join(str(tmp_path), "data")
