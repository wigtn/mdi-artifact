"""Tests for the canonical null-PI MDI (ADR-015a) and its declared smoothing (ADR-015c)."""

import math
import random

import pytest

from mdi.stats.fip import make_split
from mdi.stats.null_mdi import (
    DEFAULT_POWER,
    POWER_GRID_STEP,
    estimate_null_mdi,
    half_width,
    isotonic_non_increasing,
    mdi_at_power,
    null_mdi_entries,
    pool_inflation,
    supported_sweep,
)
from mdi.stats.synthetic import synthetic_records

ENV = "e_synthetic"
Z_975 = 1.959963985


def _gaussian_draws(*, sigma: float, n: int, seed: int) -> list[float]:
    """A seeded Gaussian sample standing in for a measured null distribution."""
    rng = random.Random(seed)
    return [rng.gauss(0.0, sigma) for _ in range(n)]


def _analytic_half_width(sigma: float, n_repeats: int, n_items: int) -> float:
    """Half-width of the null's central 95% PI for iid Gaussian scores."""
    return Z_975 * sigma * math.sqrt(2.0 / (n_repeats * n_items))


def test_null_mdi_recovers_the_analytic_half_width_for_gaussian_scores() -> None:
    """For known sigma the half-width is z_{0.975} * sigma * sqrt(2 / (N * n_items))."""
    # Given: Gaussian records with sigma = 1 over 8 items and 40 repeats (pools of 20)
    sigma = 1.0
    n_items = 8
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.5, "s_C": 1.0},
        n_items=n_items,
        n_repeats=40,
        sigma=sigma,
        seed=11,
    )
    split = make_split(screening=[], estimation=list(range(40)))
    # When: the null MDI is estimated at N = 1 and N = 5
    # Then: it matches the analytic value. N = 1 is exact in expectation; larger N
    # carries the finite-pool inflation sqrt(1 + (N-1)/pool) (pool = 20 -> ~9.5%),
    # so its tolerance is wider.
    for n_repeats, tolerance in ((1, 0.06), (5, 0.15)):
        entry = estimate_null_mdi(
            records,
            ENV,
            split=split,
            n_repeats=n_repeats,
            alphas=(0.05,),
            seed=7,
            draws_per_system=600,
            n_resamples=200,
        )[0]
        expected = _analytic_half_width(sigma, n_repeats, n_items)
        assert entry["mdi_null"] == pytest.approx(expected, rel=tolerance)
        # ... and the |delta| quantile (also recorded, per ADR-015a) sits nearby
        # because the null is symmetric
        assert entry["abs_delta_quantile"] == pytest.approx(entry["mdi_null"], rel=0.1)
        # ... in %p on likert5 the value is 100/4 times the raw one
        assert entry["mdi_null_pp"] == pytest.approx(entry["mdi_null"] * 25.0)


def test_null_mdi_ci_is_undefined_for_a_single_system_cluster() -> None:
    """One system leaves nothing to resample: the CI is None, never zero-width."""
    # Given: a single-system store
    records = synthetic_records(
        system_means={"s_A": 3.0}, n_items=4, n_repeats=10, sigma=1.0, seed=3
    )
    split = make_split(screening=[], estimation=list(range(10)))
    # When: the null MDI is estimated
    entry = estimate_null_mdi(
        records,
        ENV,
        split=split,
        n_repeats=1,
        alphas=(0.05,),
        seed=5,
        draws_per_system=100,
        n_resamples=50,
    )[0]
    # Then: the existing degenerate-cluster guard reports an undefined interval
    assert entry["mdi_null"] > 0.0
    assert entry["ci_lo"] is None
    assert entry["ci_hi"] is None
    assert entry["bootstrap"]["degenerate"] is True


def test_isotonic_smoothing_is_non_increasing_and_pools_violators() -> None:
    """PAVA: violations merge into their mean; already-monotone input is untouched."""
    # Given/When/Then
    assert isotonic_non_increasing([3.0, 1.0, 2.0, 0.5]) == [3.0, 1.5, 1.5, 0.5]
    assert isotonic_non_increasing([5.0, 4.0, 4.0, 1.0]) == [5.0, 4.0, 4.0, 1.0]
    assert isotonic_non_increasing([1.0, 2.0, 3.0]) == [2.0, 2.0, 2.0]
    assert isotonic_non_increasing([]) == []


def test_null_entries_keep_raw_values_next_to_the_smoothed_sequence() -> None:
    """The smoothing is declared, not silent: raw and smoothed both live in the entries."""
    # Given: a store supporting several budgets
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.4, "s_C": 1.0},
        n_items=5,
        n_repeats=12,
        sigma=1.0,
        seed=23,
    )
    split = make_split(screening=[0], estimation=list(range(1, 12)))
    # When: the sweep table is built
    entries = null_mdi_entries(
        records,
        split=split,
        sweep=(1, 3, 5, 8, 10),
        alphas=(0.05,),
        seed=9,
        draws_per_system=150,
        n_resamples=50,
    )
    # Then: per alpha the smoothed sequence is non-increasing in N and equals
    # PAVA over the raw sequence, which itself is preserved untouched
    raw = [entry["mdi_null"] for entry in entries]
    smoothed = [entry["mdi_null_smoothed"] for entry in entries]
    assert [entry["n_repeats"] for entry in entries] == [1, 3, 5, 8, 10]
    assert smoothed == isotonic_non_increasing(raw)
    assert all(left >= right for left, right in zip(smoothed, smoothed[1:]))
    direct = estimate_null_mdi(
        records,
        ENV,
        split=split,
        n_repeats=1,
        alphas=(0.05,),
        seed=9 + 104729,  # the sweep builder's deterministic per-(env, N) offset
        draws_per_system=150,
        n_resamples=50,
    )[0]
    assert entries[0]["mdi_null"] == direct["mdi_null"]


def test_null_sweep_is_truncated_to_the_available_estimation_repeats() -> None:
    """Budgets beyond what the estimation repeats hold are dropped, not simulated."""
    # Given: 6 repeats with repeat 0 held out -> 5 estimation repeats
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.5}, n_items=4, n_repeats=6, sigma=1.0, seed=13
    )
    split = make_split(screening=[0], estimation=list(range(1, 6)))
    # When: the default sweep is requested
    supported = supported_sweep(records, ENV, split=split, sweep=(1, 3, 5, 8, 10, 20))
    entries = null_mdi_entries(
        records,
        split=split,
        sweep=(1, 3, 5, 8, 10, 20),
        alphas=(0.05,),
        seed=2,
        draws_per_system=60,
        n_resamples=50,
    )
    # Then: only N <= 5 survives
    assert supported == [1, 3, 5]
    assert [entry["n_repeats"] for entry in entries] == [1, 3, 5]


def test_null_sweep_ignores_a_single_cell_richer_than_the_rest() -> None:
    """One outlier cell must not open a budget column the other cells can only simulate."""
    # Given: every cell holds 3 estimation repeats except one, which holds 5
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.5}, n_items=4, n_repeats=6, sigma=1.0, seed=17
    )
    split = make_split(screening=[0], estimation=list(range(1, 6)))
    outlier = (records[0]["system_id"], records[0]["item_id"])
    ragged = [
        record
        for record in records
        if record["repeat_idx"] <= 3 or (record["system_id"], record["item_id"]) == outlier
    ]

    # When: the sweep is resolved over the ragged store
    supported = supported_sweep(ragged, ENV, split=split, sweep=(1, 3, 5, 8, 10, 20))

    # Then: the shared floor of 3 governs, not the outlier's 5
    assert supported == [1, 3]


def test_null_mdi_is_deterministic_under_a_fixed_seed() -> None:
    """The same seed reproduces the table exactly (AGENTS.md §3.3)."""
    # Given: one store
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.6}, n_items=4, n_repeats=10, sigma=1.0, seed=29
    )
    split = make_split(screening=[0], estimation=list(range(1, 10)))
    # When: built twice
    kwargs = dict(
        split=split,
        sweep=(1, 5),
        alphas=(0.10, 0.05),
        seed=31,
        draws_per_system=80,
        n_resamples=50,
    )
    first = null_mdi_entries(records, **kwargs)  # type: ignore[arg-type]
    second = null_mdi_entries(records, **kwargs)  # type: ignore[arg-type]
    # Then: identical, sorted, self-describing
    assert first == second
    keys = [(e["task"], e["scale"], e["env_id"], e["n_repeats"], -e["alpha"]) for e in first]
    assert keys == sorted(keys)


def test_null_mdi_shrinks_as_the_repeat_budget_grows() -> None:
    """The paper's central claim holds for the primary path: MDI_null falls with N."""
    # Given: plenty of repeats
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.3, "s_C": 0.9},
        n_items=6,
        n_repeats=24,
        sigma=1.0,
        seed=41,
    )
    split = make_split(screening=[], estimation=list(range(24)))
    # When: the raw (unsmoothed) sequence is estimated over a wide sweep
    entries = null_mdi_entries(
        records,
        split=split,
        sweep=(1, 5, 20),
        alphas=(0.05,),
        seed=17,
        draws_per_system=300,
        n_resamples=50,
    )
    raw = [entry["mdi_null"] for entry in entries]
    # Then: strictly ordered even before smoothing at this sample size
    assert [entry["n_repeats"] for entry in entries] == [1, 5, 20]
    assert raw[0] > raw[1] > raw[2]


# --- ADR-024a: the power column ---------------------------------------------------------


def test_mdi_at_power_returns_the_threshold_when_power_is_one_half() -> None:
    """At power 0.5 the family collapses to the alpha-level value the paper tabulates."""
    # Given: a symmetric null distribution and its central-95% half-width
    draws = sorted(_gaussian_draws(sigma=1.0, n=40_000, seed=3))
    threshold = half_width(draws, 0.05, assume_sorted=True)
    # When: the power read-off is taken at exactly one half
    value = mdi_at_power(draws, threshold, 0.5, assume_sorted=True)
    # Then: it is the threshold itself — a true gain of MDI is caught half the time
    assert value == pytest.approx(threshold, rel=POWER_GRID_STEP)


def test_mdi_at_power_recovers_the_normal_multiplier_at_eighty_percent() -> None:
    """On a Gaussian null the measured factor matches 1 + z_{0.80}/z_{0.975} = 1.4294."""
    # Given: Gaussian null draws
    draws = sorted(_gaussian_draws(sigma=1.0, n=200_000, seed=5))
    threshold = half_width(draws, 0.05, assume_sorted=True)
    # When: the 80% detection point is read off them
    value = mdi_at_power(draws, threshold, 0.80, assume_sorted=True)
    # Then: the ratio reproduces the closed form the manuscript quotes
    assert value is not None
    assert value / threshold == pytest.approx(1.0 + 0.841621234 / Z_975, abs=0.01)


def test_mdi_at_power_is_undefined_when_the_grid_does_not_reach_the_target() -> None:
    """An unreachable power returns None rather than extrapolating (ADR-018 principle)."""
    # Given: Gaussian null draws and a grid capped just above the threshold
    draws = sorted(_gaussian_draws(sigma=1.0, n=20_000, seed=9))
    threshold = half_width(draws, 0.05, assume_sorted=True)
    # When: 99.9% detection is asked for inside a grid that stops at 1.1x
    value = mdi_at_power(draws, threshold, 0.999, grid_max=1.1, assume_sorted=True)
    # Then: the answer is "not measured here", not a number
    assert value is None


def test_power_column_travels_beside_the_raw_sequence_and_is_smoothed() -> None:
    """The power entries are emitted per (env, N) with their own isotonic sequence."""
    # Given: a two-budget sweep over Gaussian records
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.4, "s_C": 0.8},
        n_items=6,
        n_repeats=20,
        sigma=1.0,
        seed=23,
    )
    split = make_split(screening=[], estimation=list(range(20)))
    # When: the sweep is built
    entries = null_mdi_entries(
        records, split=split, sweep=(1, 5), alphas=(0.05,), seed=13, draws_per_system=300
    )
    # Then: every entry carries a power reading above its alpha-level value ...
    assert entries
    for entry in entries:
        assert entry["power"] == pytest.approx(DEFAULT_POWER)
        assert entry["mdi_power"] is not None
        assert entry["mdi_power"] > entry["mdi_null"]
        assert entry["ratio_power"] == pytest.approx(
            entry["mdi_power"] / entry["mdi_null"], rel=1e-9
        )
    # ... and the smoothed power sequence is non-increasing in N
    ordered = sorted(entries, key=lambda e: e["n_repeats"])
    sequence = [e["mdi_power_smoothed"] for e in ordered]
    assert all(value is not None for value in sequence)
    defined = [value for value in sequence if value is not None]
    assert all(a >= b for a, b in zip(defined, defined[1:]))


# --- ADR-025a: the finite-pool correction -----------------------------------------------


def test_pool_inflation_is_exactly_one_at_a_single_repeat() -> None:
    """At N=1 the split-half penalty and the within-pool gain cancel exactly."""
    # Given: any pool split
    # When / Then: one draw from each pool is just two draws from the r repeats
    for pool_term in (1.0 / 9 + 1.0 / 10, 1.0 / 3 + 1.0 / 4, 1.0 / 20 + 1.0 / 20):
        assert pool_inflation(pool_term, 1) == pytest.approx(1.0)


def test_pool_inflation_grows_with_the_budget_and_shrinks_with_the_pool() -> None:
    """More repeats against a fixed pool inflates more; a bigger pool inflates less."""
    # Given: the dense (9+10) and anchor (3+4) splits this project actually uses
    dense = 1.0 / 9 + 1.0 / 10
    anchor = 1.0 / 3 + 1.0 / 4
    # When / Then: monotone in N ...
    assert pool_inflation(dense, 1) < pool_inflation(dense, 5) < pool_inflation(dense, 10)
    # ... and the thinner pool is always worse at the same budget
    assert pool_inflation(anchor, 5) > pool_inflation(dense, 5)
    # ... with the magnitudes the appendix reports (dense N=10 ~ +40%)
    assert pool_inflation(dense, 10) == pytest.approx(1.396, abs=0.005)
    assert pool_inflation(anchor, 5) == pytest.approx(1.472, abs=0.005)


def test_corrected_mdi_is_independent_of_the_estimation_pool_size() -> None:
    """The corrected column estimates the environment; the raw one also carries our budget.

    The claim ADR-025a makes is *invariance*: the same environment measured with
    different numbers of stored repeats must give the same corrected threshold,
    while the raw one moves. A residual of a few percent survives because the
    correction rescales a standard deviation whereas the reported value is an
    empirical quantile, and the two differ while the item mean is not yet normal.
    """
    # Given: the same Gaussian environment measured with 8, 20 and 40 repeats
    sigma = 1.0
    n_items = 16
    raw: list[float] = []
    corrected: list[float] = []
    for n_stored in (8, 20, 40):
        records = synthetic_records(
            system_means={"s_A": 0.0, "s_B": 0.5, "s_C": 1.0},
            n_items=n_items,
            n_repeats=n_stored,
            sigma=sigma,
            seed=31,
        )
        split = make_split(screening=[], estimation=list(range(n_stored)))
        # When: MDI is estimated at a fixed budget of 5
        entry = estimate_null_mdi(
            records,
            ENV,
            split=split,
            n_repeats=5,
            alphas=(0.05,),
            seed=17,
            draws_per_system=1500,
            n_resamples=50,
        )[0]
        raw.append(entry["mdi_null"])
        corrected.append(entry["mdi_null_corrected"])

    def _spread(values: list[float]) -> float:
        return (max(values) - min(values)) / (sum(values) / len(values))

    # Then: the raw column swings with the stored pool — the artifact is large ...
    assert _spread(raw) > 0.15
    assert raw[0] > raw[-1]
    # ... the corrected column is far flatter ...
    assert _spread(corrected) < 0.12
    assert _spread(corrected) < _spread(raw) / 2.0
    # ... and each corrected value sits on the infinite-pool estimand
    expected = _analytic_half_width(sigma, 5, n_items)
    for value in corrected:
        assert value == pytest.approx(expected, rel=0.15)
