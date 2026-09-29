"""Tests for the Exp 1 variance analysis (FR-005)."""

import math
from typing import List

import pytest

from mdi.stats import records as rec
from mdi.stats.synthetic import synthetic_records
from mdi.stats.variance import (
    INTERVAL,
    NOMINAL,
    ORDINAL,
    cell_flip_rate,
    env_variance,
    item_flip_rates,
    krippendorff_alpha,
    pooled_sigma,
    rank_flip_rate,
    variance_report,
)


def test_pooled_sigma_recovers_the_injected_sigma_on_synthetic_records() -> None:
    """Pooled within-cell sigma must recover the noise injected into the fixture."""
    # Given: 20 items x 40 repeats generated with sigma = 0.8
    records = synthetic_records(
        system_means={"s_A": 3.0}, n_items=20, n_repeats=40, sigma=0.8, seed=101
    )
    cells = rec.group_by_cell(records)
    # When: sigma is pooled over all cells
    sigma = pooled_sigma([rec.cell_values(cell) for cell in cells.values()])
    # Then: it matches the injected value within Monte-Carlo error
    assert sigma is not None
    assert sigma == pytest.approx(0.8, rel=0.05)


def test_pooled_sigma_is_none_when_no_cell_has_two_repeats() -> None:
    """A single-repeat store cannot express judge noise."""
    # Given: one repeat per cell
    records = synthetic_records(
        system_means={"s_A": 3.0}, n_items=4, n_repeats=1, sigma=1.0, seed=5
    )
    cells = rec.group_by_cell(records)
    # When/Then: sigma is undefined
    assert pooled_sigma([rec.cell_values(cell) for cell in cells.values()]) is None


def test_cell_flip_rate_counts_pairwise_disagreements() -> None:
    """Flip rate is the share of distinct repeat pairs that disagree."""
    # Given: three repeats, two of which agree
    values = [4.0, 4.0, 5.0]
    # When: the flip rate is computed (pairs: 4-4, 4-5, 4-5)
    # Then: two of three pairs disagree
    assert cell_flip_rate(values) == pytest.approx(2.0 / 3.0)
    assert cell_flip_rate([4.0, 4.0, 4.0]) == 0.0
    assert cell_flip_rate([4.0]) is None


def test_item_flip_rates_cover_every_scored_cell_in_sorted_order() -> None:
    """Every (env, item, system) cell with >= 2 repeats yields one sorted row."""
    # Given: 3 items x 2 systems x 5 repeats on a rounded (Likert) scale
    records = synthetic_records(
        system_means={"s_A": 3.0, "s_B": 3.4},
        n_items=3,
        n_repeats=5,
        sigma=1.0,
        seed=9,
        round_to=1.0,
    )
    # When: item-level flip rates are computed
    rows = item_flip_rates(records)
    # Then: one row per cell, sorted, each with the full pair count
    assert len(rows) == 6
    assert rows == sorted(rows, key=lambda row: (row["env_id"], row["item_id"], row["system_id"]))
    assert all(row["n_pairs"] == 10 for row in rows)
    assert all(0.0 <= row["flip_rate"] <= 1.0 for row in rows)


def test_rank_flip_rate_is_zero_when_systems_never_overlap() -> None:
    """A separated pair must never flip on a single-run comparison."""
    # Given: two systems separated far beyond the noise
    records = synthetic_records(
        system_means={"s_A": 1.0, "s_B": 9.0}, n_items=4, n_repeats=6, sigma=0.2, seed=3
    )
    # When: the rank-flip rate is measured
    flip = rank_flip_rate(records, "e_synthetic", "s_A", "s_B")
    # Then: the reference ordering holds in every comparison
    assert flip is not None
    assert flip["reference_sign"] == -1
    assert flip["rank_flip_rate"] == 0.0
    assert flip["n_comparisons"] == 4 * 6 * 6


def test_rank_flip_rate_is_near_one_half_for_identical_systems() -> None:
    """Two systems with the same true mean flip about half the time."""
    # Given: identical true means, pure judge noise
    records = synthetic_records(
        system_means={"s_A": 3.0, "s_B": 3.0}, n_items=8, n_repeats=12, sigma=1.0, seed=77
    )
    # When: the rank-flip rate is measured
    flip = rank_flip_rate(records, "e_synthetic", "s_A", "s_B")
    # Then: it sits close to chance
    assert flip is not None
    assert flip["rank_flip_rate"] == pytest.approx(0.5, abs=0.08)


def test_krippendorff_alpha_is_one_for_perfect_agreement() -> None:
    """Zero observed disagreement gives alpha = 1."""
    # Given: two units whose repeats agree exactly
    units: List[List[float]] = [[1.0, 1.0], [3.0, 3.0]]
    # When/Then: alpha is 1 for every difference metric
    for level in (INTERVAL, NOMINAL, ORDINAL):
        assert krippendorff_alpha(units, level) == pytest.approx(1.0)


def test_krippendorff_alpha_matches_the_hand_computed_interval_value() -> None:
    """The interval-metric alpha matches the closed-form value for a 2x2 example."""
    # Given: units [1, 3] and [3, 1] -> D_o = 4, D_e = 32/12
    units: List[List[float]] = [[1.0, 3.0], [3.0, 1.0]]
    # When: alpha is computed with the interval metric
    alpha = krippendorff_alpha(units, INTERVAL)
    # Then: alpha = 1 - 4 / (8/3) = -0.5
    assert alpha == pytest.approx(-0.5)


def test_krippendorff_alpha_ordinal_equals_nominal_for_binary_values() -> None:
    """With two distinct values the ordinal metric is a positive rescaling of the nominal one."""
    # Given: a binary-valued coding
    units: List[List[float]] = [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]]
    # When: alpha is computed under both metrics
    ordinal = krippendorff_alpha(units, ORDINAL)
    nominal = krippendorff_alpha(units, NOMINAL)
    # Then: the coefficients coincide (alpha is invariant to scaling delta^2)
    assert ordinal is not None and nominal is not None
    assert ordinal == pytest.approx(nominal)


def test_krippendorff_alpha_is_none_for_a_degenerate_pool() -> None:
    """A single-valued pool has zero expected disagreement, so alpha is undefined."""
    # Given: every repeat identical everywhere
    units: List[List[float]] = [[2.0, 2.0], [2.0, 2.0]]
    # When/Then: alpha is undefined rather than infinite
    assert krippendorff_alpha(units, INTERVAL) is None


def test_krippendorff_alpha_rejects_an_unknown_level() -> None:
    """An unsupported difference metric raises rather than defaulting silently."""
    # Given/When/Then
    with pytest.raises(ValueError, match="unknown alpha level"):
        krippendorff_alpha([[1.0, 2.0]], "ratio")


def test_env_variance_reports_the_parse_failure_rate_per_env() -> None:
    """Parse failures are preserved and reported, never dropped (ADR-011 D7)."""
    # Given: 10 repeats per cell of which repeat 0 always fails to parse
    records = synthetic_records(
        system_means={"s_A": 3.0},
        n_items=5,
        n_repeats=10,
        sigma=0.5,
        seed=13,
        parse_failure_repeats=(0,),
    )
    # When: the env summary is built
    rows = env_variance(records)
    # Then: one row, 10% parse failures, sigma from the remaining repeats
    assert len(rows) == 1
    assert rows[0]["parse_failure_rate"] == pytest.approx(0.1)
    assert rows[0]["n_scores"] == 45
    assert rows[0]["sigma"] == pytest.approx(0.5, rel=0.2)


def test_variance_report_emits_every_system_pair_once() -> None:
    """Rank-flip rows cover each unordered system pair exactly once."""
    # Given: three systems
    records = synthetic_records(
        system_means={"s_A": 3.0, "s_B": 3.2, "s_C": 3.6},
        n_items=4,
        n_repeats=6,
        sigma=0.7,
        seed=21,
    )
    # When: the full FR-005 report is assembled
    report = variance_report(records)
    # Then: three pairs, sorted, with alpha reported at the interval level
    pairs = [(row["system_a"], row["system_b"]) for row in report["rank_flips"]]
    assert pairs == [("s_A", "s_B"), ("s_A", "s_C"), ("s_B", "s_C")]
    assert report["alpha_level"] == INTERVAL
    assert report["envs"][0]["krippendorff_alpha"] is not None
    assert not math.isnan(report["envs"][0]["krippendorff_alpha"])


def test_env_variance_reports_a_sigma_ci_bracketing_the_point() -> None:
    """The bootstrap CI must contain the point sigma and be a real interval."""
    # Given: 30 items x 20 repeats at a known sigma
    records = synthetic_records(
        system_means={"s_A": 3.0, "s_B": 3.1}, n_items=30, n_repeats=20, sigma=0.6, seed=202
    )
    # When: env variance is computed with the bootstrap
    (env,) = env_variance(records, seed=202, sigma_resamples=500)
    # Then: the CI brackets the point estimate and has positive width
    assert env["sigma"] is not None
    assert env["sigma_ci_lo"] is not None and env["sigma_ci_hi"] is not None
    assert env["sigma_ci_lo"] <= env["sigma"] <= env["sigma_ci_hi"]
    assert env["sigma_ci_hi"] > env["sigma_ci_lo"]
    assert env["sigma_bootstrap"] is not None
    assert env["sigma_bootstrap"]["n_resamples"] == 500


def test_sigma_ci_is_reproducible_under_a_fixed_seed() -> None:
    """Same records and seed must give a bit-identical CI (determinism, AGENTS.md 3.3)."""
    # Given: identical inputs
    records = synthetic_records(
        system_means={"s_A": 3.0}, n_items=20, n_repeats=15, sigma=0.5, seed=7
    )
    # When: env variance runs twice with the same seed
    (a,) = env_variance(records, seed=99, sigma_resamples=300)
    (b,) = env_variance(records, seed=99, sigma_resamples=300)
    # Then: the intervals match exactly
    assert (a["sigma_ci_lo"], a["sigma_ci_hi"]) == (b["sigma_ci_lo"], b["sigma_ci_hi"])


def test_sigma_ci_is_undefined_with_a_single_item() -> None:
    """One item leaves the item-cluster bootstrap nothing to resample."""
    # Given: a single item scored many times
    records = synthetic_records(
        system_means={"s_A": 3.0}, n_items=1, n_repeats=20, sigma=0.5, seed=3
    )
    # When: env variance is computed
    (env,) = env_variance(records, seed=3, sigma_resamples=200)
    # Then: sigma is still estimated but its CI is undefined, not zero-width
    assert env["sigma"] is not None
    assert env["sigma_ci_lo"] is None and env["sigma_ci_hi"] is None
