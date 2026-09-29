"""Tests for the Exp 3 decay fit (FR-007): sigma^2/N + omega^2 recovery."""

from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from mdi.stats.decay import (
    DEFAULT_REPEAT_SWEEP,
    common_groups,
    decay_report,
    fit_decay,
    omega_components,
    pair_gap_shift,
)
from mdi.stats.synthetic import population_variance, synthetic_records

ENV = "e_synthetic"


def test_decay_sweep_includes_n_eight_per_adr_013() -> None:
    """The repeat sweep carries N=8 (ADR-013, first-author decision 2026-07-31 #5)."""
    # Given/When/Then: the default sweep is the ADR-011 D1 set plus 8
    assert DEFAULT_REPEAT_SWEEP == (1, 3, 5, 8, 10, 20)


def test_decay_fit_recovers_a_known_sigma_squared_and_omega_squared() -> None:
    """With fixed procedural offsets, the fit recovers the injected variance components."""
    # Given: sigma = 0.6 within groups and two fixed group offsets -> omega^2 = 0.25
    sigma = 0.6
    offsets = (-0.5, 0.5)
    records = synthetic_records(
        system_means={"s_A": 3.0},
        n_items=12,
        n_repeats=80,
        sigma=sigma,
        seed=5,
        group_offsets=offsets,
    )
    # When: the decay model is fitted from bootstrap slices of the full-N cells
    fit = fit_decay(records, ENV, seed=3, draws_per_cell=600)
    # Then: slope and intercept match the injected components
    assert fit["sigma_sq"] == pytest.approx(sigma**2, rel=0.1)
    assert fit["omega_sq"] == pytest.approx(population_variance(offsets), rel=0.1)
    assert fit["omega_sq_debiased"] == pytest.approx(population_variance(offsets), rel=0.1)
    assert fit["r_squared"] is not None and fit["r_squared"] > 0.99
    assert fit["icc"] == pytest.approx(0.25 / (0.25 + sigma**2), rel=0.15)


def test_decay_reports_a_bootstrap_ci_on_the_noise_floor() -> None:
    """omega^2 carries a cell-cluster bootstrap CI (needed by the ADR-015c display rule)."""
    # Given: a real noise floor (omega^2 = 0.25) over many cells
    records = synthetic_records(
        system_means={"s_A": 3.0},
        n_items=10,
        n_repeats=40,
        sigma=0.6,
        seed=31,
        group_offsets=(-0.5, 0.5),
    )
    # When: the decay model is fitted
    fit = fit_decay(records, ENV, seed=2, draws_per_cell=300, n_resamples=200)
    # Then: the interval brackets the point estimate and excludes zero
    assert fit["omega_sq_ci_lo"] is not None and fit["omega_sq_ci_hi"] is not None
    assert fit["omega_sq_ci_lo"] <= fit["omega_sq"] <= fit["omega_sq_ci_hi"]
    assert fit["omega_sq_ci_lo"] > 0.0
    assert fit["omega_ci_hi"] == pytest.approx(fit["omega_sq_ci_hi"] ** 0.5)
    assert fit["bootstrap"]["n_clusters"] == fit["n_cells"]


def test_decay_noise_floor_ci_touches_zero_without_a_procedural_axis() -> None:
    """With a single paraphrase group the clamped intercept's CI lower bound is zero."""
    # Given: one procedural group only
    records = synthetic_records(
        system_means={"s_A": 3.0}, n_items=8, n_repeats=20, sigma=1.0, seed=11
    )
    # When: the decay model is fitted
    fit = fit_decay(records, ENV, seed=3, draws_per_cell=200, n_resamples=200)
    # Then: the floor is below detection — the display rule's trigger condition
    assert fit["omega_sq"] == pytest.approx(0.0, abs=0.02)
    assert fit["omega_sq_ci_lo"] is not None
    assert fit["omega_sq_ci_lo"] == pytest.approx(0.0, abs=1e-12)


def test_decay_noise_floor_is_zero_without_a_procedural_axis() -> None:
    """A single procedural group leaves nothing that repeat-averaging cannot remove."""
    # Given: one paraphrase group only
    records = synthetic_records(
        system_means={"s_A": 3.0}, n_items=12, n_repeats=40, sigma=1.0, seed=5
    )
    # When: the decay model is fitted
    fit = fit_decay(records, ENV, seed=3, draws_per_cell=400)
    # Then: omega^2 collapses to ~0 while sigma^2 is recovered
    assert fit["omega_sq"] == pytest.approx(0.0, abs=0.02)
    assert fit["sigma_sq"] == pytest.approx(1.0, rel=0.1)
    assert fit["mean_groups_per_cell"] == 1.0


def test_decay_measured_spread_sits_above_the_theoretical_line_when_a_floor_exists() -> None:
    """The deviation from 1/sqrt(N) grows with N exactly because of the noise floor."""
    # Given: a real noise floor
    records = synthetic_records(
        system_means={"s_A": 3.0},
        n_items=10,
        n_repeats=40,
        sigma=0.8,
        seed=31,
        group_offsets=(-0.4, 0.4),
    )
    # When: the decay curve is measured
    fit = fit_decay(records, ENV, seed=2, draws_per_cell=500)
    deviations = [point["deviation"] for point in fit["points"]]
    # Then: the N=1 point anchors the reference and later points sit above it
    assert fit["points"][0]["n"] == 1
    assert deviations[0] == pytest.approx(0.0, abs=1e-9)
    assert all(left <= right + 1e-9 for left, right in zip(deviations, deviations[1:])), deviations
    assert deviations[-1] > 0.1


def test_decay_uses_only_the_sweep_values_the_data_supports() -> None:
    """Budgets larger than the available repeats per group are dropped, not extrapolated."""
    # Given: 12 repeats split across two groups -> 6 per group
    records = synthetic_records(
        system_means={"s_A": 3.0},
        n_items=6,
        n_repeats=12,
        sigma=1.0,
        seed=7,
        group_offsets=(-0.3, 0.3),
    )
    # When: the default sweep {1,3,5,8,10,20} is requested
    fit = fit_decay(records, ENV, seed=1, draws_per_cell=200)
    # Then: only N <= 6 survives
    assert fit["sweep"] == [1, 3, 5]
    assert [point["n"] for point in fit["points"]] == [1, 3, 5]


def test_decay_raises_when_fewer_than_two_sweep_points_survive() -> None:
    """A fit needs at least two distinct budgets."""
    # Given: two repeats per group only
    records = synthetic_records(
        system_means={"s_A": 3.0}, n_items=4, n_repeats=2, sigma=1.0, seed=7
    )
    # When/Then: the fit refuses rather than inventing a line
    with pytest.raises(ValueError, match=">= 2 distinct N values"):
        fit_decay(records, ENV, seed=1, sweep=(1, 20), draws_per_cell=50)


def test_decay_fit_is_reproducible_under_a_fixed_seed() -> None:
    """The same seed reproduces the fit exactly."""
    # Given: one fixture
    records = synthetic_records(
        system_means={"s_A": 3.0},
        n_items=6,
        n_repeats=20,
        sigma=1.0,
        seed=19,
        group_offsets=(-0.4, 0.4),
    )
    # When: fitted twice with the same seed
    first = fit_decay(records, ENV, seed=8, draws_per_cell=200)
    second = fit_decay(records, ENV, seed=8, draws_per_cell=200)
    # Then: identical output, metadata included
    assert first == second


def test_decay_report_covers_every_env_in_sorted_order() -> None:
    """One fit per environment, sorted by env_id."""
    # Given: two environments in one store
    records = synthetic_records(
        system_means={"s_A": 3.0},
        n_items=5,
        n_repeats=20,
        sigma=1.0,
        seed=23,
        env_id="e_bbb",
        group_offsets=(-0.2, 0.2),
    ) + synthetic_records(
        system_means={"s_A": 3.0},
        n_items=5,
        n_repeats=20,
        sigma=0.5,
        seed=24,
        env_id="e_aaa",
        group_offsets=(-0.2, 0.2),
    )
    # When: the report is assembled
    report = decay_report(records, seed=4, draws_per_cell=150)
    # Then: sorted envs with distinct sigmas
    assert [fit["env_id"] for fit in report["fits"]] == ["e_aaa", "e_bbb"]
    assert report["fits"][0]["sigma"] < report["fits"][1]["sigma"]


# --------------------------------------------------------------------------------------
# ADR-020: main-effect / interaction split of the between-group floor
# --------------------------------------------------------------------------------------


def _keyed(shift: float, tilt: float) -> List[Tuple[str, Dict[str, Dict[str, float]]]]:
    """Two items x three systems x two paraphrases with a known main/interaction split.

    ``shift`` moves every system in the item together (main effect); ``tilt``
    moves them in opposite directions (interaction). Paraphrase 0 gets ``-x``
    and paraphrase 1 ``+x``, so the population variance of a component of
    amplitude ``x`` is exactly ``x**2``.
    """
    out = {}
    for item in ("i1", "i2"):
        systems = {}
        for index, system in enumerate(("s1", "s2", "s3")):
            sign = 1.0 if index == 0 else (-1.0 if index == 1 else 0.0)
            systems[system] = {
                "p0": -shift - sign * tilt,
                "p1": shift + sign * tilt,
            }
        out[item] = systems
    return list(out.items())


def _components(
    items: Sequence[Tuple[str, Dict[str, Dict[str, float]]]],
    groups_override: Optional[Sequence[str]] = None,
) -> Tuple[float, float]:
    """omega_components, asserting it identified the split -- narrows Optional for mypy."""
    parts = omega_components(items, groups_override)
    assert parts is not None, "the decomposition should be identified for this fixture"
    return parts


def test_omega_components_recovers_a_pure_main_effect() -> None:
    """A shift shared by every system is the part that cancels in an A-vs-B difference."""
    # Given: a common shift of 0.2 and no system-specific tilt
    items = _keyed(shift=0.2, tilt=0.0)
    # When: the floor is decomposed
    main, interaction = _components(items)
    # Then: all of it lands in the main component
    assert main == pytest.approx(0.04, abs=1e-9)
    assert interaction == pytest.approx(0.0, abs=1e-9)


def test_omega_components_recovers_a_pure_interaction() -> None:
    """A tilt that moves systems apart is the part that survives the difference."""
    # Given: no common shift, a system-specific tilt of 0.3
    items = _keyed(shift=0.0, tilt=0.3)
    # When: decomposed
    main, interaction = _components(items)
    # Then: nothing is common, and the interaction carries the variance
    assert main == pytest.approx(0.0, abs=1e-9)
    assert interaction > 0.0


def test_omega_components_add_up_to_the_between_group_variance() -> None:
    """main + interaction reconstructs the between-group floor exactly (ADR-020 §a)."""
    # Given: both components present
    items = _keyed(shift=0.2, tilt=0.3)
    # When: decomposed, and the raw between-group variance computed directly
    main, interaction = _components(items)
    direct = []
    for _item, systems in items:
        for _system, groups in systems.items():
            values = [groups[g] for g in sorted(groups)]
            centre = sum(values) / len(values)
            direct.append(sum((v - centre) ** 2 for v in values) / len(values))
    between = sum(direct) / len(direct)
    # Then: the split is exact, not approximate
    assert main + interaction == pytest.approx(between, rel=1e-9)


def test_omega_components_returns_none_without_two_systems() -> None:
    """One system per item identifies no split: the estimator declines rather than guesses."""
    # Given: a single system
    items = [("i1", {"s1": {"p0": 0.1, "p1": 0.3}})]
    # When/Then: no decomposition is reported
    assert omega_components(items) is None


def test_common_groups_intersects_only_identifiable_cells() -> None:
    """The cluster list is the label set every usable cell carries (ADR-023)."""
    # Given: one identifiable item over p0/p1 and a single-system item carrying a stray p9
    items = [
        ("i1", {"s1": {"p0": 0.1, "p1": 0.3}, "s2": {"p0": 0.2, "p1": 0.4}}),
        ("i2", {"s1": {"p0": 0.1, "p9": 0.3}}),
    ]
    # When: the cluster labels are collected
    # Then: the unidentifiable item contributes nothing, so p9 is not a cluster
    assert common_groups(items) == ["p0", "p1"]


def test_omega_components_override_restricts_to_the_requested_labels() -> None:
    """Passing a label subset decomposes over that subset only (ADR-023 resample path)."""
    # Given: three paraphrases where p2 carries a large extra main shift
    items = [
        (
            "i1",
            {
                "s1": {"p0": -0.2, "p1": 0.2, "p2": 5.0},
                "s2": {"p0": -0.2, "p1": 0.2, "p2": 5.0},
            },
        )
    ]
    # When: decomposed over p0/p1 only
    main, interaction = _components(items, ["p0", "p1"])
    # Then: the excluded outlier does not enter -- variance of (-0.2, +0.2) is 0.04
    assert main == pytest.approx(0.04, abs=1e-9)
    assert interaction == pytest.approx(0.0, abs=1e-9)


def test_omega_components_override_repeats_a_label_like_a_bootstrap_draw() -> None:
    """A duplicated label is a legal resample and shrinks the spread (ADR-023)."""
    # Given: a pure main effect of amplitude 0.2 over two paraphrases
    items = _keyed(shift=0.2, tilt=0.0)
    # When: p0 is drawn twice, as a with-replacement resample can do
    main, _interaction = _components(items, ["p0", "p0", "p1"])
    # Then: the mean moves toward p0, so the population variance drops below 0.04
    assert 0.0 < main < 0.04


def test_omega_components_override_skips_items_missing_a_label() -> None:
    """An item that lacks a requested label is dropped, never silently re-based."""
    # Given: one item with p0/p1 and one with p0 only
    items = [
        ("i1", {"s1": {"p0": -0.2, "p1": 0.2}, "s2": {"p0": -0.2, "p1": 0.2}}),
        ("i2", {"s1": {"p0": 1.0, "p5": 9.0}, "s2": {"p0": 1.0, "p5": 9.0}}),
    ]
    # When: decomposed over p0/p1
    main, _interaction = _components(items, ["p0", "p1"])
    # Then: only the first item contributes -- p5's huge offset never leaks in
    assert main == pytest.approx(0.04, abs=1e-9)


def test_group_cluster_ci_is_wider_than_the_item_cluster_ci() -> None:
    """Resampling 2 paraphrases carries more uncertainty than resampling many cells."""
    # Given: a synthetic env whose floor comes from fixed per-group offsets
    records = synthetic_records(
        system_means={"s1": 3.0, "s2": 3.4},
        n_items=12,
        n_repeats=8,
        sigma=0.6,
        seed=7,
        group_offsets=(-0.5, 0.5),
    )
    # When: fitted
    fit = fit_decay(records, ENV, seed=3, draws_per_cell=300, n_resamples=200)
    # Then: the group-clustered interval on the sum is reported over 2 clusters
    assert fit["n_group_clusters"] == 2
    assert fit["omega_sq_between_group_ci_lo"] is not None
    assert fit["omega_sq_between_group_ci_hi"] is not None
    item_lo, item_hi = fit["omega_sq_main_ci_lo"], fit["omega_sq_main_ci_hi"]
    group_lo, group_hi = fit["omega_sq_main_group_ci_lo"], fit["omega_sq_main_group_ci_hi"]
    assert item_lo is not None and item_hi is not None
    assert group_lo is not None and group_hi is not None
    item_width = item_hi - item_lo
    group_width = group_hi - group_lo
    assert group_width >= item_width


def test_pair_gap_shift_is_zero_without_a_system_specific_effect() -> None:
    """A prompt shift common to every system leaves the A-vs-B gap alone."""
    # Given: a pure main effect -- both systems move together in each group
    items = [
        ("i1", {"s1": {"p0": 3.0, "p1": 3.5}, "s2": {"p0": 2.0, "p1": 2.5}}),
        ("i2", {"s1": {"p0": 3.2, "p1": 3.7}, "s2": {"p0": 2.2, "p1": 2.7}}),
    ]
    # When: the pair gap is tracked across groups
    result = pair_gap_shift(items)
    # Then: the gap is 1.0 under either prompt, so it does not travel
    assert result is not None
    median_sd, _max_sd, n_pairs, flips = result
    assert median_sd == pytest.approx(0.0, abs=1e-12)
    assert n_pairs == 1
    assert flips == 0


def test_pair_gap_shift_measures_a_system_specific_swing() -> None:
    """A prompt that favours one system moves the gap by a measurable amount."""
    # Given: p1 lifts s1 by 0.4 but leaves s2 alone -> gaps of 1.0 and 1.4
    items = [
        ("i1", {"s1": {"p0": 3.0, "p1": 3.4}, "s2": {"p0": 2.0, "p1": 2.0}}),
        ("i2", {"s1": {"p0": 3.0, "p1": 3.4}, "s2": {"p0": 2.0, "p1": 2.0}}),
    ]
    # When: measured
    result = pair_gap_shift(items)
    # Then: the population SD of (1.0, 1.4) is 0.2
    assert result is not None
    assert result[0] == pytest.approx(0.2, abs=1e-12)
    assert result[3] == 0


def test_pair_gap_shift_flags_a_prompt_induced_sign_change() -> None:
    """A gap that changes sign across prompts is counted, not averaged away."""
    # Given: s1 leads under p0 and trails under p1
    items = [("i1", {"s1": {"p0": 3.0, "p1": 2.0}, "s2": {"p0": 2.0, "p1": 3.0}})]
    # When: measured
    result = pair_gap_shift(items)
    # Then: the one pair is reported as a sign flip
    assert result is not None
    assert result[3] == 1


def test_pair_gap_shift_averages_over_items_before_differencing() -> None:
    """Item-specific swings cancel in the average; only the common part survives."""
    # Given: p1 favours s1 in i1 and s2 in i2 by the same amount
    items = [
        ("i1", {"s1": {"p0": 3.0, "p1": 3.6}, "s2": {"p0": 2.0, "p1": 2.0}}),
        ("i2", {"s1": {"p0": 3.0, "p1": 3.0}, "s2": {"p0": 2.0, "p1": 2.6}}),
    ]
    # When: measured over the item average
    result = pair_gap_shift(items)
    # Then: the two item-level swings cancel, so the averaged gap does not move
    assert result is not None
    assert result[0] == pytest.approx(0.0, abs=1e-12)


def test_pair_gap_shift_median_averages_the_middle_two_of_an_even_count() -> None:
    """Four systems make six pairs, so the median takes the even branch (ADR-020).

    The production number comes from this branch: reporting the upper median
    instead once turned 1.15 into 1.28. Every other test here uses two systems,
    where the two conventions agree, so the mistake could recur unseen.
    """
    # Given: four systems whose six pair gaps travel by known, distinct amounts.
    # Systems sit at 0, a, b, c under p0 and shift by 0, x, y, z under p1, so a
    # pair's gap moves by |shift difference| / 2 in population SD over two groups.
    offsets = {"s1": 0.0, "s2": 0.2, "s3": 0.6, "s4": 1.4}
    items = [
        (
            "i1",
            {system: {"p0": 0.0, "p1": shift} for system, shift in offsets.items()},
        )
    ]
    # When: measured
    result = pair_gap_shift(items)
    # Then: the six pairwise SDs are half the offset differences
    assert result is not None
    median_sd, max_sd, n_pairs, _flips = result
    expected = sorted(
        abs(offsets[a] - offsets[b]) / 2.0
        for index, a in enumerate(sorted(offsets))
        for b in sorted(offsets)[index + 1 :]
    )
    assert n_pairs == 6
    assert max_sd == pytest.approx(expected[-1], abs=1e-12)
    # the two middle values differ, so the upper-median convention would fail here
    assert expected[2] != expected[3]
    assert median_sd == pytest.approx((expected[2] + expected[3]) / 2.0, abs=1e-12)
