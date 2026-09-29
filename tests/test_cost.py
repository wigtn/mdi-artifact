"""Tests for the dry-run cost estimator, budget caps and the spend ledger (FR-004)."""

import os
from typing import Any, Dict, List

import pytest

from mdi.cost import (
    DEFAULT_CUMULATIVE_CAP_USD,
    DEFAULT_PILOT_CAP_USD,
    VERDICT_BLOCK,
    VERDICT_OK,
    CallSpec,
    UnknownPriceError,
    append_ledger_entry,
    budget_verdict,
    caps,
    estimate_cost,
    estimate_tokens,
    ledger_total_usd,
    price_call,
    price_table,
    read_ledger,
)

CONFIG: Dict[str, Any] = {
    "pricing": {
        "judge-a": {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 2.0},
        "judge-b": {"input_usd_per_mtok": 10.0, "output_usd_per_mtok": 20.0},
    },
    "budget": {"pilot_cap_usd": 5.0, "cumulative_cap_usd": 10.0},
}


def calls(model: str, count: int, in_tokens: int = 1000, out_tokens: int = 100) -> List[CallSpec]:
    """A homogeneous list of planned calls."""
    return [CallSpec(model=model, in_tokens=in_tokens, out_tokens=out_tokens) for _ in range(count)]


def test_caps_fall_back_to_the_prd_defaults_when_the_config_is_silent() -> None:
    """PRD §4.2 hard caps apply unless the config raises them."""
    # Given: a config with no budget block
    # When / Then
    assert caps({}) == {
        "pilot_cap_usd": DEFAULT_PILOT_CAP_USD,
        "cumulative_cap_usd": DEFAULT_CUMULATIVE_CAP_USD,
    }


def test_estimate_tokens_is_deterministic_for_the_same_text() -> None:
    """The dry-run token heuristic is a pure function of the rendered prompt."""
    # Given: a prompt
    prompt = "x" * 400
    # When / Then: 400/4 + overhead, stable across calls
    assert estimate_tokens(prompt) == estimate_tokens(prompt) == 108


def test_price_call_uses_the_config_declared_unit_price() -> None:
    """Unit prices come from the config, never from code (FR-004)."""
    # Given: the declared table
    table = price_table(CONFIG)
    # When: one call is priced
    cost = price_call(table, "judge-a", 1_000_000, 1_000_000)
    # Then: input + output rates apply per million tokens
    assert cost == pytest.approx(3.0)


def test_price_call_raises_when_the_model_has_no_declared_price() -> None:
    """An unpriced model is an estimation error, never a silent zero."""
    # Given / When / Then
    with pytest.raises(UnknownPriceError, match="no price declared"):
        price_call(price_table(CONFIG), "judge-unknown", 10, 10)


def test_price_table_raises_when_an_entry_is_incomplete() -> None:
    """A half-declared price is rejected at read time."""
    # Given: a table missing the output rate
    config = {"pricing": {"judge-a": {"input_usd_per_mtok": 1.0}}}
    # When / Then
    with pytest.raises(ValueError, match="output_usd_per_mtok"):
        price_table(config)


def test_estimate_cost_rolls_up_totals_per_model() -> None:
    """The estimate reports call/token totals and a per-model breakdown."""
    # Given: 10 calls on judge-a and 5 on judge-b
    planned = calls("judge-a", 10) + calls("judge-b", 5)
    # When: estimated
    estimate = estimate_cost(CONFIG, planned)
    # Then: totals and a sorted per-model rollup
    assert estimate["calls"] == 15
    assert estimate["in_tokens"] == 15_000
    assert estimate["out_tokens"] == 1_500
    assert [entry["model"] for entry in estimate["by_model"]] == ["judge-a", "judge-b"]
    assert estimate["by_model"][0]["cost_usd"] == pytest.approx(10 * (1000 * 1 + 100 * 2) / 1e6)
    assert estimate["cost_usd"] == pytest.approx(
        estimate["by_model"][0]["cost_usd"] + estimate["by_model"][1]["cost_usd"]
    )


def test_budget_verdict_is_ok_when_the_projection_fits_under_both_caps() -> None:
    """A cheap run passes the gate."""
    # Given / When
    verdict = budget_verdict(CONFIG, projected_usd=1.0, spent_usd=2.0, pilot=True)
    # Then
    assert verdict["verdict"] == VERDICT_OK
    assert verdict["slice_cap_usd"] == 5.0


def test_budget_verdict_blocks_when_the_pilot_cap_is_exceeded() -> None:
    """The pilot cap bounds a single pilot slice."""
    # Given / When
    verdict = budget_verdict(CONFIG, projected_usd=6.0, spent_usd=0.0, pilot=True)
    # Then
    assert verdict["verdict"] == VERDICT_BLOCK
    assert "pilot cap" in verdict["reason"]


def test_budget_verdict_blocks_when_the_ledger_plus_projection_passes_the_cumulative_cap() -> None:
    """Prior spend counts: the cumulative cap is on the ledger, not on one run."""
    # Given: $9.50 already spent against a $10 cumulative cap
    # When: another $1 is projected
    verdict = budget_verdict(CONFIG, projected_usd=1.0, spent_usd=9.5, pilot=True)
    # Then
    assert verdict["verdict"] == VERDICT_BLOCK
    assert "cumulative cap" in verdict["reason"]


def test_ledger_accumulates_and_totals_across_appended_entries(tmp_path: Any) -> None:
    """The ledger is append-only and its total is the running spend (PRD §4.2)."""
    # Given: an empty ledger path
    path = os.path.join(str(tmp_path), "data", "ledger.jsonl")
    assert ledger_total_usd(path) == 0.0
    # When: three run entries are appended
    append_ledger_entry(path, {"run_id": "r_1", "experiment": "e1", "cost_usd": 0.25})
    append_ledger_entry(path, {"run_id": "r_2", "experiment": "e1", "cost_usd": 0.5})
    append_ledger_entry(path, {"run_id": "r_3", "experiment": "e2", "cost_usd": 1.0})
    # Then: totals accumulate, and can be scoped to one experiment
    assert ledger_total_usd(path) == pytest.approx(1.75)
    assert ledger_total_usd(path, experiment="e1") == pytest.approx(0.75)
    assert [entry["run_id"] for entry in read_ledger(path)] == ["r_1", "r_2", "r_3"]


def test_append_ledger_entry_raises_when_cost_is_missing(tmp_path: Any) -> None:
    """A ledger line without a cost is not a ledger line."""
    # Given / When / Then
    path = os.path.join(str(tmp_path), "ledger.jsonl")
    with pytest.raises(ValueError, match="cost_usd"):
        append_ledger_entry(path, {"run_id": "r_1"})


def test_read_ledger_skips_a_truncated_final_line(tmp_path: Any) -> None:
    """A crash mid-append loses one line, not the whole ledger."""
    # Given: a ledger whose last line was cut off
    path = os.path.join(str(tmp_path), "ledger.jsonl")
    append_ledger_entry(path, {"run_id": "r_1", "cost_usd": 0.25})
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"run_id": "r_2", "cost_us')
    # When / Then: the intact entry survives
    assert [entry["run_id"] for entry in read_ledger(path)] == ["r_1"]
    assert ledger_total_usd(path) == pytest.approx(0.25)
