"""Synthetic score records with known ground truth — validation only, never data.

Used by the analysis tests and by VALIDATE-stage notebooks to check that the
estimators recover injected parameters (sigma, omega, a true delta). Records
produced here are **not** measurements: they carry ``run_id`` values prefixed
``r_synthetic`` and zero cost/usage, they are never written to ``data/raw/``
(AGENTS.md §3.1 — that path belongs to the runner alone), and no number derived
from them may enter the paper (AGENTS.md §3.6).

Generation is fully seeded and deterministic. The procedural axis is modelled by
*fixed* group offsets assigned round-robin to ``repeat_idx``, so the population
variance of the offsets is exactly the ``omega^2`` the decay fit should recover.
"""

import math
import random
from typing import Dict, List, Mapping, Optional, Sequence

from mdi.store import SCHEMA_VERSION, ScoreRecord, Usage

SYNTHETIC_RUN_ID: str = "r_synthetic"
SYNTHETIC_TS: str = "2026-01-01T00:00:00+00:00"


def population_variance(values: Sequence[float]) -> float:
    """Population variance (1/n divisor) — the exact omega^2 of a fixed offset set."""
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / len(values)


def make_record(
    *,
    env_id: str,
    item_id: str,
    system_id: str,
    repeat_idx: int,
    score: Optional[float],
    paraphrase_id: str,
    task: str,
    scale: str,
    benchmark: str = "synthetic",
    judge_model: str = "synthetic/judge",
    judge_model_version: str = "v0",
    temperature: float = 1.0,
    seed: int = 0,
    run_id: str = SYNTHETIC_RUN_ID,
) -> ScoreRecord:
    """Build one schema-v2 record; ``score=None`` marks a parse failure (ADR-011 D7)."""
    return ScoreRecord(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        env_id=env_id,
        judge_model=judge_model,
        judge_model_version=judge_model_version,
        judge_tier="synthetic",
        serving_engine=None,
        quantization=None,
        seed=seed,
        prompt_id="p_synthetic",
        paraphrase_id=paraphrase_id,
        temperature=temperature,
        scale=scale,
        task=task,
        benchmark=benchmark,
        item_id=item_id,
        system_id=system_id,
        repeat_idx=repeat_idx,
        request_payload_hash="sha256:" + "0" * 64,
        raw_response="" if score is None else f"score: {score}",
        parsed_score=score,
        parse_ok=score is not None,
        usage=Usage(in_tokens=0, out_tokens=0),
        cost_usd=0.0,
        ts=SYNTHETIC_TS,
    )


def synthetic_records(
    *,
    system_means: Mapping[str, float],
    n_items: int,
    n_repeats: int,
    sigma: float,
    seed: int,
    env_id: str = "e_synthetic",
    task: str = "summarization",
    scale: str = "likert5",
    group_offsets: Sequence[float] = (0.0,),
    item_effect_sd: float = 0.0,
    parse_failure_repeats: Sequence[int] = (),
    round_to: Optional[float] = None,
    item_prefix: str = "i_",
) -> List[ScoreRecord]:
    """Generate one environment's records with known sigma, omega and system deltas.

    ``score = system_mean + item_effect + group_offset + Normal(0, sigma)``.
    Repeat ``r`` belongs to procedural group ``r % len(group_offsets)`` (exposed
    as ``paraphrase_id``), so ``omega^2`` equals
    :func:`population_variance` of *group_offsets* by construction.
    """
    if n_items < 1 or n_repeats < 1:
        raise ValueError("n_items and n_repeats must be >= 1")
    if not group_offsets:
        raise ValueError("group_offsets must be non-empty")

    rng = random.Random(seed)
    item_ids = [f"{item_prefix}{index:04d}" for index in range(n_items)]
    item_effects: Dict[str, float] = {
        item_id: (rng.gauss(0.0, item_effect_sd) if item_effect_sd > 0 else 0.0)
        for item_id in item_ids
    }
    failures = set(parse_failure_repeats)

    records: List[ScoreRecord] = []
    for item_id in item_ids:
        for system_id in sorted(system_means):
            for repeat_idx in range(n_repeats):
                group = repeat_idx % len(group_offsets)
                if repeat_idx in failures:
                    score: Optional[float] = None
                else:
                    value = (
                        system_means[system_id]
                        + item_effects[item_id]
                        + group_offsets[group]
                        + rng.gauss(0.0, sigma)
                    )
                    if round_to is not None and round_to > 0:
                        value = round(value / round_to) * round_to
                    score = value
                records.append(
                    make_record(
                        env_id=env_id,
                        item_id=item_id,
                        system_id=system_id,
                        repeat_idx=repeat_idx,
                        score=score,
                        paraphrase_id=f"pp_{group}",
                        task=task,
                        scale=scale,
                        seed=seed,
                    )
                )
    return records
