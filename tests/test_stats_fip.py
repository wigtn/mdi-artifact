"""Tests for FIP estimation (FR-006) — the ADR-007 guard and analytic recovery."""

import math
from typing import List, Sequence

import pytest

from mdi.stats import records as rec
from mdi.stats.fip import (
    OverlappingRepeatsError,
    RepeatSplit,
    bin_edges_from_sigma,
    conversion_table,
    estimate_fip,
    fip_report,
    make_split,
    validate_split,
)
from mdi.stats.synthetic import synthetic_records
from mdi.stats.variance import pooled_sigma
from mdi.store import ScoreRecord

ENV = "e_synthetic"


def _normal_cdf(value: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _empirical_delta(records: Sequence[ScoreRecord], system_a: str, system_b: str) -> float:
    """Mean over items of the two systems' all-repeat mean difference."""
    cells = rec.group_by_cell(records)
    items = sorted({key.item_id for key in cells if key.system_id == system_a})
    deltas: List[float] = []
    for item_id in items:
        values_a = rec.cell_values(cells[rec.CellKey(ENV, item_id, system_a)])
        values_b = rec.cell_values(cells[rec.CellKey(ENV, item_id, system_b)])
        deltas.append(sum(values_a) / len(values_a) - sum(values_b) / len(values_b))
    return sum(deltas) / len(deltas)


def _analytic_reversal(delta: float, sigma: float, n_repeats: int, n_items: int) -> float:
    """P(the re-evaluation disagrees in sign) for i.i.d. Normal scores.

    With independent observation and re-evaluation of equal budget, the reversal
    probability is ``2p(1-p)`` where ``p = Phi(delta / sd)`` and
    ``sd = sqrt(2 sigma^2 / (N * items))``.
    """
    sd = math.sqrt(2.0 * sigma * sigma / (n_repeats * n_items))
    p = _normal_cdf(delta / sd)
    return 2.0 * p * (1.0 - p)


@pytest.mark.parametrize("n_repeats", [1, 5])
def test_fip_recovers_the_analytic_reversal_rate_for_a_known_sigma_and_delta(
    n_repeats: int,
) -> None:
    """On synthetic data with known sigma and delta, FIP matches the closed form."""
    # Given: one pair separated by delta = 0.5 with sigma = 1.0 over 8 items
    records = synthetic_records(
        system_means={"s_A": 0.5, "s_B": 0.0}, n_items=8, n_repeats=60, sigma=1.0, seed=11
    )
    split = make_split(screening=[], estimation=list(range(60)))
    cells = rec.group_by_cell(records)
    sigma = pooled_sigma([rec.cell_values(cell) for cell in cells.values()])
    assert sigma is not None
    delta = _empirical_delta(records, "s_A", "s_B")

    # When: FIP is estimated nonparametrically from the measured repeats
    curve = estimate_fip(
        records, ENV, split=split, n_repeats=n_repeats, seed=7, draws_per_pair=1500, n_resamples=60
    )
    draws = sum(entry["n_draws"] for entry in curve["bins"])
    reversals = sum(entry["n_reversals"] for entry in curve["bins"])

    # Then: the measured reversal rate matches the analytic rate at the store's
    # own (delta, sigma) — the estimator resamples the empirical distribution
    expected = _analytic_reversal(delta, sigma, n_repeats, 8)
    assert draws == 1500
    assert reversals / draws == pytest.approx(expected, abs=0.03)


def test_fip_falls_as_the_observed_delta_grows() -> None:
    """The curve must be decreasing in the observed improvement."""
    # Given: several pairs spanning a range of true deltas
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.15, "s_C": 0.4, "s_D": 0.8, "s_E": 1.4, "s_F": 2.2},
        n_items=6,
        n_repeats=24,
        sigma=1.0,
        seed=17,
    )
    split = make_split(screening=[], estimation=list(range(24)))
    # When: a single-run FIP curve is estimated
    curve = estimate_fip(records, ENV, split=split, n_repeats=1, seed=5, draws_per_pair=150)
    populated = [entry for entry in curve["bins"] if entry["n_draws"] >= 30]
    values = [entry["fip"] for entry in populated]
    # Then: every successive bin has a lower flip probability
    assert len(populated) >= 3
    assert all(
        left is not None and right is not None and left >= right
        for left, right in zip(values, values[1:])
    )


def test_fip_raises_when_screening_and_estimation_repeats_overlap() -> None:
    """ADR-007: estimating FIP on the repeats that selected the pairs is forbidden."""
    # Given: a split whose screening repeat also appears in the estimation set
    records = synthetic_records(
        system_means={"s_A": 0.3, "s_B": 0.0}, n_items=4, n_repeats=8, sigma=1.0, seed=2
    )
    leaky = RepeatSplit(screening=[0, 1], estimation=[1, 2, 3, 4])
    # When/Then: both the guard and the estimator refuse
    with pytest.raises(OverlappingRepeatsError, match="ADR-007"):
        validate_split(leaky)
    with pytest.raises(OverlappingRepeatsError, match="overlap at \\[1\\]"):
        estimate_fip(records, ENV, split=leaky, n_repeats=1, seed=1, draws_per_pair=10)
    with pytest.raises(OverlappingRepeatsError):
        make_split(screening=[3], estimation=[3, 4])


def test_fip_excludes_the_screening_repeats_from_the_estimate() -> None:
    """Held-out screening repeats must not appear in the estimation pool."""
    # Given: repeat 0 held out for screening
    records = synthetic_records(
        system_means={"s_A": 0.4, "s_B": 0.0}, n_items=4, n_repeats=10, sigma=1.0, seed=8
    )
    split = make_split(screening=[0], estimation=list(range(1, 10)))
    # When: FIP is estimated
    curve = estimate_fip(records, ENV, split=split, n_repeats=1, seed=4, draws_per_pair=50)
    # Then: the recorded provenance shows the disjoint split
    assert curve["meta"]["screening_repeats"] == [0]
    assert curve["meta"]["estimation_repeats"] == list(range(1, 10))
    assert (
        set(curve["meta"]["screening_repeats"]) & set(curve["meta"]["estimation_repeats"]) == set()
    )


def test_fip_raises_when_fewer_than_two_estimation_repeats_remain() -> None:
    """Disjoint observation / re-evaluation pools need at least two repeats."""
    # Given/When/Then
    with pytest.raises(ValueError, match="at least two estimation repeats"):
        make_split(screening=[0], estimation=[1])


def test_fip_is_reproducible_under_a_fixed_seed() -> None:
    """Identical seed and inputs produce an identical curve."""
    # Given: one fixture and one seed
    records = synthetic_records(
        system_means={"s_A": 0.4, "s_B": 0.0}, n_items=4, n_repeats=12, sigma=1.0, seed=6
    )
    split = make_split(screening=[0], estimation=list(range(1, 12)))
    # When: the same estimate runs twice
    first = estimate_fip(records, ENV, split=split, n_repeats=1, seed=99, draws_per_pair=60)
    second = estimate_fip(records, ENV, split=split, n_repeats=1, seed=99, draws_per_pair=60)
    # Then: the curves are equal, metadata included
    assert first == second


def test_bin_edges_follow_the_adr_007_sigma_multiples() -> None:
    """Default bin edges are the ADR-007 screening bins in score units."""
    # Given: sigma = 0.5
    # When: edges are derived from the default multiples
    edges = bin_edges_from_sigma(0.5)
    # Then: 0, 0.5, 1, 2, 4 sigma
    assert edges == [0.0, 0.25, 0.5, 1.0, 2.0]
    with pytest.raises(ValueError, match="sigma must be positive"):
        bin_edges_from_sigma(0.0)


def test_conversion_table_states_the_reversal_rate_per_delta_bin() -> None:
    """The conversion table renders "a single-run delta = x reverses y% of the time"."""
    # Given: a measured single-run curve
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.3, "s_C": 0.9},
        n_items=5,
        n_repeats=16,
        sigma=1.0,
        seed=4,
    )
    split = make_split(screening=[0], estimation=list(range(1, 16)))
    curve = estimate_fip(records, ENV, split=split, n_repeats=1, seed=12, draws_per_pair=120)
    # When: the conversion table is built
    rows = conversion_table(curve)
    # Then: every populated bin yields one statement with a percentage
    assert rows
    assert all(row["n_draws"] > 0 for row in rows)
    assert all(row["reversal_pct"] is not None for row in rows)
    assert rows[0]["statement"].startswith("a single-run delta = ")
    assert "reverses" in rows[0]["statement"]


def test_fip_report_covers_every_requested_budget() -> None:
    """One curve per (env, N) is produced, sorted and self-describing."""
    # Given: a store and a two-point budget sweep
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.5}, n_items=4, n_repeats=12, sigma=1.0, seed=15
    )
    split = make_split(screening=[0], estimation=list(range(1, 12)))
    # When: the report is assembled
    report = fip_report(
        records, split=split, n_repeats_sweep=(1, 5), seed=3, draws_per_pair=60, n_resamples=50
    )
    # Then: two curves, both carrying their bootstrap metadata
    assert [curve["n_repeats"] for curve in report["curves"]] == [1, 5]
    assert all(curve["meta"]["bootstrap"]["seed"] >= 3 for curve in report["curves"])
    assert report["conversion_table"]
