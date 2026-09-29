"""Tests for the MDI lookup table (FR-009) — MDI as a read-off of the FIP curve."""

import math
from typing import Dict, List, Optional, cast

import pytest

from mdi.stats.fip import FipBin, FipCurve, FipMeta, estimate_fip, make_split
from mdi.stats.mdi_table import (
    BIN_UPPER,
    DEFAULT_ALPHAS,
    build_mdi_table,
    mdi_from_curve,
)
from mdi.stats.null_mdi import NullMdiEntry
from mdi.stats.synthetic import synthetic_records

ENV = "e_synthetic"


def _curve(points: List[FipBin], n_repeats: int = 1, sigma: float = 1.0) -> FipCurve:
    """Wrap explicit bins into a curve so the read-off rule can be tested in isolation."""
    return FipCurve(
        env_id=ENV,
        task="summarization",
        scale="likert5",
        n_repeats=n_repeats,
        bins=points,
        meta=FipMeta(
            n_repeats=n_repeats,
            sigma=sigma,
            bin_edges=[entry["lo"] for entry in points],
            sigma_bin_multiples=[0.0, 0.5, 1.0, 2.0, 4.0],
            draws_per_pair=100,
            n_pairs=1,
            n_items=4,
            n_draws=sum(entry["n_draws"] for entry in points),
            n_zero_delta_draws=0,
            screening_repeats=[0],
            estimation_repeats=[1, 2, 3],
            tie_policy="tie_counts_as_reversal",
            bootstrap={"seed": 1, "n_resamples": 10, "ci_level": 0.95, "method": "percentile"},
        ),
    )


def _bin(lo: float, hi: Optional[float], delta: float, fip: float, draws: int = 100) -> FipBin:
    """Build one populated FIP bin."""
    return FipBin(
        lo=lo,
        hi=hi,
        mean_delta=delta,
        n_draws=draws,
        n_reversals=int(round(fip * draws)),
        fip=fip,
        ci_lo=max(0.0, fip - 0.02),
        ci_hi=min(1.0, fip + 0.02),
    )


def test_mdi_interpolates_between_the_bins_that_bracket_alpha() -> None:
    """The default read-off interpolates linearly between bin representatives."""
    # Given: FIP 0.20 at delta 1.0 and FIP 0.00 at delta 2.0
    curve = _curve([_bin(0.0, 1.5, 1.0, 0.20), _bin(1.5, None, 2.0, 0.00)])
    # When: MDI is read off at alpha = 0.05
    entry = mdi_from_curve(curve, alpha=0.05)
    # Then: MDI = 1.0 + (0.20-0.05)/(0.20-0.00) * 1.0 = 1.75
    assert entry["attained"] is True
    assert entry["mdi"] == pytest.approx(1.75)
    assert entry["mdi_in_sigma"] == pytest.approx(1.75)


def test_mdi_reports_the_bin_upper_edge_under_the_conservative_readoff() -> None:
    """``bin_upper`` reports the crossing bin's upper edge instead of interpolating."""
    # Given: the same curve
    curve = _curve([_bin(0.0, 1.5, 1.0, 0.20), _bin(1.5, None, 2.0, 0.00)])
    # When: the conservative rule is used
    entry = mdi_from_curve(curve, alpha=0.05, readoff=BIN_UPPER)
    # Then: the crossing bin has no upper edge, so its representative delta is reported
    assert entry["mdi"] == pytest.approx(2.0)
    assert entry["readoff"] == BIN_UPPER


def test_mdi_is_not_attained_when_no_bin_reaches_alpha() -> None:
    """An environment coarser than every probed delta reports no MDI, not a guess."""
    # Given: a curve that never drops below 0.2
    curve = _curve([_bin(0.0, 1.0, 0.5, 0.40), _bin(1.0, None, 1.5, 0.22)])
    # When: MDI is read off at alpha = 0.05
    entry = mdi_from_curve(curve, alpha=0.05)
    # Then: the entry is explicit about non-attainment
    assert entry["attained"] is False
    assert entry["mdi"] is None
    assert entry["mdi_in_sigma"] is None


def test_mdi_flags_a_curve_that_is_already_below_alpha_everywhere() -> None:
    """When even the smallest measured delta is safe, MDI sits below the measured range."""
    # Given: a curve entirely under alpha
    curve = _curve([_bin(0.0, 1.0, 0.4, 0.01), _bin(1.0, None, 1.5, 0.00)])
    # When: MDI is read off
    entry = mdi_from_curve(curve, alpha=0.05)
    # Then: the smallest measured delta is reported and flagged
    assert entry["mdi"] == pytest.approx(0.4)
    assert entry["below_measured_range"] is True


def test_mdi_ignores_a_single_noisy_dip_below_alpha() -> None:
    """The crossing must be sustained: a lone dip cannot set MDI."""
    # Given: a dip at delta 0.5 followed by a bin back above alpha
    curve = _curve(
        [
            _bin(0.0, 0.4, 0.2, 0.30),
            _bin(0.4, 0.8, 0.5, 0.02),
            _bin(0.8, 1.6, 1.0, 0.12),
            _bin(1.6, None, 2.0, 0.01),
        ]
    )
    # When: MDI is read off
    entry = mdi_from_curve(curve, alpha=0.05)
    # Then: the read-off uses the last sustained crossing, not the dip
    assert entry["mdi"] is not None
    assert entry["mdi"] > 1.0


def test_mdi_ignores_bins_with_too_few_draws() -> None:
    """Sparse bins are not trusted for the read-off."""
    # Given: a low-FIP bin backed by 3 draws only
    curve = _curve([_bin(0.0, 1.0, 0.5, 0.40, draws=200), _bin(1.0, None, 1.5, 0.0, draws=3)])
    # When: MDI is read off with the default minimum
    entry = mdi_from_curve(curve, alpha=0.05)
    # Then: the sparse bin is excluded and MDI is not attained
    assert entry["n_bins_used"] == 1
    assert entry["attained"] is False


def test_mdi_rejects_an_alpha_outside_the_unit_interval() -> None:
    """Alpha must be a probability."""
    # Given/When/Then
    curve = _curve([_bin(0.0, None, 1.0, 0.1)])
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        mdi_from_curve(curve, alpha=1.5)


def test_mdi_decreases_monotonically_as_the_repeat_budget_grows() -> None:
    """The paper's central claim: more repeats resolve smaller improvements."""
    # Given: pairs spanning a range of true deltas in one environment
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.15, "s_C": 0.4, "s_D": 0.8, "s_E": 1.4, "s_F": 2.2},
        n_items=6,
        n_repeats=24,
        sigma=1.0,
        seed=17,
    )
    split = make_split(screening=[], estimation=list(range(24)))
    # When: the lookup table is built over N = 1, 5, 20
    table = build_mdi_table(
        records,
        split=split,
        sweep=(1, 5, 20),
        alphas=DEFAULT_ALPHAS,
        seed=99,
        draws_per_pair=150,
        n_resamples=100,
    )
    by_alpha: Dict[float, List[float]] = {}
    for entry in table["entries"]:
        # An unattained MDI is "coarser than anything measured" -> +inf keeps the order total
        value = entry["mdi"] if entry["mdi"] is not None else math.inf
        by_alpha.setdefault(entry["alpha"], []).append(value)

    # Then: at every error level, MDI is non-increasing in N
    assert sorted(by_alpha) == [0.01, 0.05, 0.10]
    for alpha, values in sorted(by_alpha.items()):
        assert len(values) == 3
        assert all(left >= right for left, right in zip(values, values[1:])), (
            f"alpha={alpha}: MDI not monotone in N: {values}"
        )
    # ... and a tighter alpha never yields a smaller MDI at the same budget
    for index in range(3):
        assert by_alpha[0.10][index] <= by_alpha[0.05][index] <= by_alpha[0.01][index]


def test_mdi_table_rows_are_sorted_and_carry_their_readoff_parameters() -> None:
    """Rows are deterministic and self-describing for the appendix sensitivity table."""
    # Given: a small store
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.5, "s_C": 1.2},
        n_items=5,
        n_repeats=16,
        sigma=1.0,
        seed=6,
    )
    split = make_split(screening=[0], estimation=list(range(1, 16)))
    # When: the table is built twice with the same seed
    kwargs = dict(split=split, sweep=(1, 5), alphas=(0.05,), seed=21, draws_per_pair=80)
    first = build_mdi_table(records, n_resamples=50, **kwargs)  # type: ignore[arg-type]
    second = build_mdi_table(records, n_resamples=50, **kwargs)  # type: ignore[arg-type]
    # Then: identical, sorted, and carrying the read-off parameters
    assert first == second
    keys = [
        (row["task"], row["scale"], row["env_id"], row["n_repeats"]) for row in first["entries"]
    ]
    assert keys == sorted(keys)
    assert first["readoff"] == "interpolate"
    assert first["min_bin_draws"] == 30


def test_mdi_table_emits_both_paths_and_their_consistency_block() -> None:
    """ADR-015: the null-PI path is primary, the FIP crossing validates it, nothing hides."""
    # Given: a small store
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.5, "s_C": 1.2},
        n_items=5,
        n_repeats=16,
        sigma=1.0,
        seed=6,
    )
    split = make_split(screening=[0], estimation=list(range(1, 16)))
    # When: the table is built
    table = build_mdi_table(
        records,
        split=split,
        sweep=(1, 5),
        alphas=(0.05,),
        seed=21,
        draws_per_pair=80,
        draws_per_system=100,
        n_resamples=50,
    )
    # Then: both paths are declared and present
    assert table["primary"] == "null_pi_half_width"
    assert table["validation"] == "fip_crossing"
    null_keys = {(e["env_id"], e["n_repeats"], e["alpha"]) for e in table["null_entries"]}
    fip_keys = {(e["env_id"], e["n_repeats"], e["alpha"]) for e in table["entries"]}
    assert null_keys == {(ENV, 1, 0.05), (ENV, 5, 0.05)}
    # ... and the consistency block covers the union of keys with honest arithmetic
    blocks = table["consistency"]
    assert {(b["env_id"], b["n_repeats"], b["alpha"]) for b in blocks} == null_keys | fip_keys
    for block in blocks:
        if block["mdi_null"] is not None and block["mdi_fip_crossing"] is not None:
            assert block["abs_diff"] == pytest.approx(
                abs(block["mdi_null"] - block["mdi_fip_crossing"])
            )
            assert block["ratio"] == pytest.approx(block["mdi_fip_crossing"] / block["mdi_null"])
        else:
            assert block["ratio"] is None
            assert block["abs_diff"] is None


def test_mdi_read_off_agrees_with_the_curve_it_came_from() -> None:
    """The table entry and a direct read-off of the same curve cannot disagree."""
    # Given: one measured curve
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.6, "s_C": 1.5},
        n_items=5,
        n_repeats=16,
        sigma=1.0,
        seed=9,
    )
    split = make_split(screening=[0], estimation=list(range(1, 16)))
    curve = estimate_fip(records, ENV, split=split, n_repeats=1, seed=33, draws_per_pair=120)
    # When: the same curve is read off directly and through the table builder
    direct = mdi_from_curve(curve, alpha=0.05)
    table = build_mdi_table(
        records, split=split, sweep=(1,), alphas=(0.05,), seed=33, draws_per_pair=120
    )
    # Then: the values match
    assert table["entries"][0]["mdi"] == direct["mdi"]


def test_reverse_entries_recommend_the_smallest_sufficient_n() -> None:
    """The reverse table returns the first sweep budget whose MDI meets the target."""
    from mdi.stats.mdi_table import reverse_entries_from_null

    # Given: a null table where MDI shrinks with N (2.0, 1.0, 0.5 %p at N=1,5,10)
    null_entries = [
        {
            "env_id": "e1",
            "task": "t",
            "scale": "likert5",
            "n_repeats": n,
            "alpha": 0.05,
            "mdi_null": 0.0,
            "mdi_null_pp": pp,
            "mdi_null_smoothed": 0.0,
            "mdi_null_smoothed_pp": pp,
        }
        for n, pp in [(1, 2.0), (5, 1.0), (10, 0.5)]
    ]
    # When: reverse entries are built for targets 1.5, 0.8, 0.3 %p
    rev = reverse_entries_from_null(
        cast(List[NullMdiEntry], null_entries), targets_pp=[1.5, 0.8, 0.3], alpha=0.05
    )
    got = {e["target_pp"]: (e["recommended_n"], e["reachable"]) for e in rev}
    # Then: 1.5 met at N=5, 0.8 at N=10, 0.3 unreachable in this sweep
    assert got[1.5] == (5, True)
    assert got[0.8] == (10, True)
    assert got[0.3] == (None, False)


def test_reverse_entries_ignore_other_alpha_levels() -> None:
    """Only entries at the requested alpha feed the reverse table."""
    from mdi.stats.mdi_table import reverse_entries_from_null

    # Given: the same env at two alpha levels
    null_entries = [
        {
            "env_id": "e1",
            "task": "t",
            "scale": "likert5",
            "n_repeats": 1,
            "alpha": a,
            "mdi_null": 0.0,
            "mdi_null_pp": pp,
            "mdi_null_smoothed": 0.0,
            "mdi_null_smoothed_pp": pp,
        }
        for a, pp in [(0.05, 3.0), (0.10, 1.0)]
    ]
    # When: the reverse table is built at alpha=0.10
    rev = reverse_entries_from_null(
        cast(List[NullMdiEntry], null_entries), targets_pp=[1.5], alpha=0.10
    )
    # Then: only the alpha=0.10 row (1.0 <= 1.5) is used, so N=1 is recommended
    assert len(rev) == 1
    assert rev[0]["recommended_n"] == 1
