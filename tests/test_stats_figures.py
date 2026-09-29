"""Tests for deterministic figure rendering (FR-014, PRD §4.1) and the ADR-015c rules."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from mdi.stats.decay import DecayFit, fit_decay
from mdi.stats.figures import (
    FIGURE_EPOCH,
    decay_caption,
    noise_floor_display,
    promotion_stopping_title,
    render_decay_curve,
    render_fip_curve,
    render_fip_overlay,
    render_mdi_curve,
    render_mdi_overlay,
    render_promotion_stopping,
)
from mdi.stats.fip import FipCurve, estimate_fip, make_split
from mdi.stats.mdi_table import mdi_from_curve
from mdi.stats.null_mdi import NullMdiEntry, null_mdi_entries
from mdi.stats.raster import raster_diff
from mdi.stats.synthetic import synthetic_records

ENV = "e_synthetic"


def _decay_fit(
    *,
    omega_sq: float,
    omega_sq_ci_lo: Optional[float],
    omega_sq_ci_hi: Optional[float],
    mean_groups_per_cell: float = 1.0,
) -> DecayFit:
    """A minimal, fully populated DecayFit for testing the display rules in isolation."""
    return DecayFit(
        env_id=ENV,
        task="summarization",
        scale="likert5",
        sweep=[1, 5],
        points=[],
        sigma_sq=0.09,
        sigma=0.3,
        omega_sq=omega_sq,
        omega=omega_sq**0.5,
        omega_sq_raw=omega_sq,
        omega_sq_debiased=omega_sq,
        omega_sq_ci_lo=omega_sq_ci_lo,
        omega_sq_ci_hi=omega_sq_ci_hi,
        omega_ci_hi=None if omega_sq_ci_hi is None else max(omega_sq_ci_hi, 0.0) ** 0.5,
        icc=None,
        r_squared=None,
        omega_sq_main=None,
        omega_sq_int=None,
        omega_sq_between=None,
        omega_sq_main_ci_lo=None,
        omega_sq_main_ci_hi=None,
        omega_sq_int_ci_lo=None,
        omega_sq_int_ci_hi=None,
        omega_sq_main_group_ci_lo=None,
        omega_sq_main_group_ci_hi=None,
        omega_sq_int_group_ci_lo=None,
        omega_sq_int_group_ci_hi=None,
        omega_sq_between_group_ci_lo=None,
        omega_sq_between_group_ci_hi=None,
        n_group_clusters=0,
        pair_gap_shift_sd=None,
        pair_gap_shift_sd_max=None,
        pair_gap_shift_ci_lo=None,
        pair_gap_shift_ci_hi=None,
        n_pairs=0,
        pair_gap_sign_flips=None,
        group_field="paraphrase_id",
        mean_groups_per_cell=mean_groups_per_cell,
        mean_repeats_per_group=10.0,
        n_cells=8,
        draws_per_cell=100,
        bootstrap={"seed": 1, "n_resamples": 50, "ci_level": 0.95, "method": "percentile"},
    )


@pytest.fixture(scope="module")
def curve() -> FipCurve:
    """A measured single-run FIP curve over several close pairs."""
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.3, "s_C": 0.9, "s_D": 1.8},
        n_items=5,
        n_repeats=16,
        sigma=1.0,
        seed=41,
    )
    split = make_split(screening=[0], estimation=list(range(1, 16)))
    return estimate_fip(records, ENV, split=split, n_repeats=1, seed=13, draws_per_pair=120)


@pytest.fixture(scope="module")
def fit() -> DecayFit:
    """A fitted decay curve with a real noise floor."""
    records = synthetic_records(
        system_means={"s_A": 3.0},
        n_items=6,
        n_repeats=20,
        sigma=0.8,
        seed=43,
        group_offsets=(-0.4, 0.4),
    )
    return fit_decay(records, ENV, seed=13, draws_per_cell=200)


def test_fip_figure_bytes_are_identical_across_two_renders(curve: FipCurve, tmp_path: Path) -> None:
    """Byte-identical output is a hard requirement for `mdi report all` (AGENTS.md §3.3)."""
    # Given: one measured curve and two output directories
    first_dir = str(tmp_path / "first")
    second_dir = str(tmp_path / "second")
    # When: the same figure is rendered twice
    first = render_fip_curve(curve, first_dir, provenance="exp t · env e_synthetic")
    second = render_fip_curve(curve, second_dir, provenance="exp t · env e_synthetic")
    # Then: both formats match byte for byte
    assert [os.path.basename(path) for path in first] == [os.path.basename(path) for path in second]
    for left, right in zip(first, second):
        assert Path(left).read_bytes() == Path(right).read_bytes(), os.path.basename(left)


def test_decay_figure_bytes_are_identical_across_two_renders(fit: DecayFit, tmp_path: Path) -> None:
    """The decay figure is equally pinned."""
    # Given: one fitted decay model
    first_dir = str(tmp_path / "first")
    second_dir = str(tmp_path / "second")
    # When: rendered twice
    first = render_decay_curve(fit, first_dir, provenance="exp t · env e_synthetic")
    second = render_decay_curve(fit, second_dir, provenance="exp t · env e_synthetic")
    # Then: identical bytes
    for left, right in zip(first, second):
        assert Path(left).read_bytes() == Path(right).read_bytes(), os.path.basename(left)


def test_figure_files_carry_no_wall_clock_timestamp(curve: FipCurve, tmp_path: Path) -> None:
    """The PDF creation date is the pinned epoch, not the render time."""
    # Given: a rendered PDF
    paths = render_fip_curve(curve, str(tmp_path), formats=("pdf",))
    payload = Path(paths[0]).read_bytes()
    # When/Then: it carries the pinned 2025-01-01 date and the fixed producer
    assert b"/CreationDate (D:20250101000000Z)" in payload
    assert b"/Producer (mdi)" in payload
    assert b"Matplotlib" not in payload
    assert FIGURE_EPOCH == 1735689600


def test_figure_render_restores_the_ambient_source_date_epoch(
    curve: FipCurve, tmp_path: Path
) -> None:
    """Rendering must not leak its pinned environment into the caller's process."""
    # Given: no ambient SOURCE_DATE_EPOCH
    before = os.environ.get("SOURCE_DATE_EPOCH")
    # When: a figure is rendered
    render_fip_curve(curve, str(tmp_path), formats=("png",))
    # Then: the variable is restored exactly
    assert os.environ.get("SOURCE_DATE_EPOCH") == before


def test_fip_figure_marks_the_mdi_read_off_from_the_same_curve(
    curve: FipCurve, tmp_path: Path
) -> None:
    """The figure's MDI marker comes from the same read-off as the lookup table."""
    # Given: the read-off of the plotted curve
    entry = mdi_from_curve(curve, alpha=0.05)
    # When: the figure is rendered with and without an explicit entry
    default_paths = render_fip_curve(curve, str(tmp_path / "a"), formats=("png",))
    explicit_paths = render_fip_curve(curve, str(tmp_path / "b"), formats=("png",), mdi_entry=entry)
    # Then: identical bytes — the default path reads off the same value
    assert Path(default_paths[0]).read_bytes() == Path(explicit_paths[0]).read_bytes()


@pytest.fixture(scope="module")
def null_entries() -> List[NullMdiEntry]:
    """A null-PI MDI(N) series for one (env, alpha=0.05)."""
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.3, "s_C": 0.9, "s_D": 1.8},
        n_items=5,
        n_repeats=16,
        sigma=1.0,
        seed=41,
    )
    split = make_split(screening=[0], estimation=list(range(1, 16)))
    return null_mdi_entries(
        records,
        split=split,
        sweep=(1, 3, 5, 10),
        alphas=(0.05,),
        seed=13,
        draws_per_system=120,
        n_resamples=60,
    )


def test_mdi_curve_bytes_are_identical_across_two_renders(
    null_entries: List[NullMdiEntry], tmp_path: Path
) -> None:
    """The new MDI-vs-N figure (ADR-015c rule b) is pinned like every other figure."""
    # Given: one null-MDI series and two output directories
    first = render_mdi_curve(null_entries, str(tmp_path / "a"), provenance="exp t · env e")
    second = render_mdi_curve(null_entries, str(tmp_path / "b"), provenance="exp t · env e")
    # When/Then: byte-identical output in every format
    for left, right in zip(first, second):
        assert Path(left).read_bytes() == Path(right).read_bytes(), os.path.basename(left)


def test_mdi_curve_refuses_a_mixed_env_or_alpha_series(
    null_entries: List[NullMdiEntry], tmp_path: Path
) -> None:
    """One figure draws one (env, alpha) series — mixing them silently would mislead."""
    # Given: a series with a foreign alpha injected
    corrupted = list(null_entries)
    foreign = dict(null_entries[0])
    foreign["alpha"] = 0.01
    corrupted.append(foreign)  # type: ignore[arg-type]
    # When/Then: rendering refuses
    with pytest.raises(ValueError, match="one \\(env, alpha\\) series"):
        render_mdi_curve(corrupted, str(tmp_path))


def test_fip_figure_with_a_null_mdi_marker_is_deterministic(
    curve: FipCurve, tmp_path: Path
) -> None:
    """The primary-path marker (null-PI value, rule c) renders byte-identically."""
    # Given: an explicit null-PI MDI in raw score units
    first = render_fip_curve(curve, str(tmp_path / "a"), formats=("png",), null_mdi=0.8)
    second = render_fip_curve(curve, str(tmp_path / "b"), formats=("png",), null_mdi=0.8)
    without = render_fip_curve(curve, str(tmp_path / "c"), formats=("png",))
    # When/Then: identical with the same input, different when the marker changes
    assert Path(first[0]).read_bytes() == Path(second[0]).read_bytes()
    assert Path(first[0]).read_bytes() != Path(without[0]).read_bytes()


def test_noise_floor_display_never_prints_a_zero_omega() -> None:
    """Rule a: an undetected floor renders as 'below detection', never 'omega = 0.000'."""
    # Given: a floor whose point estimate is zero with a CI upper bound of 0.0016
    display = noise_floor_display(
        _decay_fit(omega_sq=0.0, omega_sq_ci_lo=0.0, omega_sq_ci_hi=0.0016)
    )
    # Then: the below-detection wording with the CI upper bound in %p (0.04/4*100 = 1 %p)
    assert display == "noise floor: below detection at this scale (≤ 1.00 %p)"
    assert "0.000" not in display

    # Given: a positive point estimate whose CI lower bound still touches zero
    display = noise_floor_display(
        _decay_fit(omega_sq=0.01, omega_sq_ci_lo=0.0, omega_sq_ci_hi=0.04)
    )
    # Then: still below detection — the data cannot exclude a zero floor
    assert display.startswith("noise floor: below detection at this scale")

    # Given: no CI at all (degenerate single-cell fit)
    display = noise_floor_display(
        _decay_fit(omega_sq=0.0, omega_sq_ci_lo=None, omega_sq_ci_hi=None)
    )
    # Then: below detection without a bound
    assert display == "noise floor: below detection at this scale"


def test_noise_floor_display_quotes_a_detected_floor_in_pp() -> None:
    """Rule a + c: a floor whose CI excludes zero is quoted as a value in %p."""
    # Given: omega^2 = 0.04 (omega = 0.2 -> 5 %p on likert5) with CI (0.01, 0.09)
    display = noise_floor_display(
        _decay_fit(omega_sq=0.04, omega_sq_ci_lo=0.01, omega_sq_ci_hi=0.09)
    )
    # Then: the omega value in %p, never a bare raw-scale zero
    assert display == "noise floor ω = 5 %p"


def test_decay_caption_states_the_measured_axis() -> None:
    """Rule a: single-paraphrase runs carry the intrinsic-axis caption verbatim."""
    # Given/When/Then: one procedural group -> the mandated sentence
    single = decay_caption(_decay_fit(omega_sq=0.0, omega_sq_ci_lo=0.0, omega_sq_ci_hi=0.001))
    assert single == (
        "This run measures the intrinsic axis only (single paraphrase); a noise floor is "
        "not expected by design and becomes measurable only with the procedural axis (Exp 3)."
    )
    # ... and a multi-group fit states the procedural axis instead
    multi = decay_caption(
        _decay_fit(
            omega_sq=0.04, omega_sq_ci_lo=0.01, omega_sq_ci_hi=0.09, mean_groups_per_cell=2.0
        )
    )
    assert multi.startswith("Procedural axis included")


def test_decay_caption_names_the_axis_not_the_record_field() -> None:
    """A caption is read by a person holding the paper, not by the schema."""
    # Given: a multi-group fit on the default procedural axis
    fit = _decay_fit(
        omega_sq=0.04, omega_sq_ci_lo=0.01, omega_sq_ci_hi=0.09, mean_groups_per_cell=2.0
    )
    # When/Then: the caption names the axis, and the storage field never appears
    caption = decay_caption(fit)
    assert "prompt-paraphrase groups per cell" in caption
    assert "paraphrase_id" not in caption


def test_decay_caption_falls_back_to_the_field_name_when_unmapped() -> None:
    """An unmapped axis still says which axis it is, rather than saying nothing."""
    # Given: a grouping field with no reader-facing label
    fit = _decay_fit(
        omega_sq=0.04, omega_sq_ci_lo=0.01, omega_sq_ci_hi=0.09, mean_groups_per_cell=2.0
    )
    fit["group_field"] = "rubric_variant"
    # When/Then: the raw field name carries the caption
    assert "rubric_variant groups per cell" in decay_caption(fit)


def test_promotion_stopping_title_counts_the_band_without_implying_an_order() -> None:
    """The title reads the lower panel; it never narrates a trajectory."""
    # Given: seed 101's verdicts, in loop order (promotions 1, 2, 5, 6 fail the band)
    clears = [False, False, True, True, False, False]
    # When/Then: the count is stated, and no ordering word appears
    title = promotion_stopping_title(101, clears)
    assert title == "Seed 101: 4 of 6 accepted promotions sit inside the noise band"
    for ordering_word in ("drift", "peak", "later", "ends", "ending"):
        assert ordering_word not in title


def test_promotion_stopping_title_handles_a_seed_that_clears_everything() -> None:
    """A seed with no band failures still gets a true title, not an empty one."""
    # Given/When/Then: every decision clears
    assert promotion_stopping_title(7, [True]) == (
        "Seed 7: 0 of 1 accepted promotions sit inside the noise band"
    )


def test_fip_figure_raises_when_no_bin_is_populated(tmp_path: Path) -> None:
    """An empty curve is a caller error, not a blank figure."""
    # Given: a curve whose bins hold no draws
    empty = dict(
        env_id=ENV,
        task="t",
        scale="s",
        n_repeats=1,
        bins=[
            {
                "lo": 0.0,
                "hi": None,
                "mean_delta": None,
                "n_draws": 0,
                "n_reversals": 0,
                "fip": None,
                "ci_lo": None,
                "ci_hi": None,
            }
        ],
        meta={"sigma": 1.0, "n_draws": 0},
    )
    # When/Then: rendering refuses
    with pytest.raises(ValueError, match="no populated bin"):
        render_fip_curve(empty, str(tmp_path))  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# Overlay + promotion renderers (2026-08-06 figure pass)
# --------------------------------------------------------------------------------------


def _promotion_rows() -> List[Dict[str, Any]]:
    """One seed's promotion trace: two decisions clear the band, one does not."""
    return [
        {"promo_num": 1, "confirm_delta": 0.05, "reversal_prob": 0.02, "clears_noise_band": True},
        {"promo_num": 2, "confirm_delta": 0.09, "reversal_prob": 0.01, "clears_noise_band": True},
        {"promo_num": 3, "confirm_delta": 0.01, "reversal_prob": 0.50, "clears_noise_band": False},
    ]


def _stopping(halt: Optional[int], peak: Optional[float]) -> Dict[str, Any]:
    """The matching stopping record; *halt*/*peak* are None when nothing clears the band."""
    return {
        "seed": 101,
        "halt_promo_num": halt,
        "peak_confirm_delta": peak,
        "final_confirm_delta": 0.01,
    }


def test_promotion_figure_bytes_are_identical_across_two_renders(tmp_path: Path) -> None:
    """The Exp 4 figure had no generator at all before this pass — it is pinned now."""
    # Given: one seed's trace
    rows, stop = _promotion_rows(), _stopping(2, 0.09)
    # When: rendered twice into separate directories
    first = render_promotion_stopping(rows, stop, str(tmp_path / "a"), stem="s")
    second = render_promotion_stopping(rows, stop, str(tmp_path / "b"), stem="s")
    # Then: byte-identical in every format (AGENTS.md §3.3)
    for left, right in zip(first, second):
        assert Path(left).read_bytes() == Path(right).read_bytes(), os.path.basename(left)


def test_promotion_figure_handles_a_seed_where_nothing_clears_the_band(tmp_path: Path) -> None:
    """Seeds 11 and 23 have no halt point; the renderer must not divide by a missing peak."""
    # Given: a stopping record with no halt and no peak
    rows, stop = _promotion_rows(), _stopping(None, None)
    # When: rendered
    paths = render_promotion_stopping(rows, stop, str(tmp_path), stem="none", formats=("png",))
    # Then: a file is produced rather than an exception
    assert Path(paths[0]).exists()


def test_promotion_figure_refuses_an_empty_trace(tmp_path: Path) -> None:
    """An empty row set is a caller bug, not an empty plot."""
    with pytest.raises(ValueError, match="no promotion rows"):
        render_promotion_stopping([], _stopping(1, 0.05), str(tmp_path), stem="empty")


def test_raster_comparator_calls_a_known_different_render_pair_different(tmp_path: Path) -> None:
    """The ADR-031 comparator must flag a real figure change -- proven on every run.

    The 8/25 procedure restored five genuinely-changed figures because the
    comparison in use could only ever answer "identical" (getbbox reads only
    alpha on RGBA). Keeping a known-different pair in the suite means a
    comparator that cannot fail cannot pass the tests either.
    """
    # Given: the same trace rendered with a halt and with no halt
    rows = _promotion_rows()
    with_halt = render_promotion_stopping(
        rows, _stopping(2, 0.09), str(tmp_path / "a"), stem="s", formats=("png",)
    )
    no_halt = render_promotion_stopping(
        rows, _stopping(None, None), str(tmp_path / "b"), stem="s", formats=("png",)
    )
    # When: comparing the pair, and one file against itself
    across = raster_diff(with_halt[0], no_halt[0])
    same = raster_diff(with_halt[0], with_halt[0])
    # Then: the known difference is seen; the identity case stays identical
    assert across.identical is False
    assert across.n_pixels_differing > 0
    assert same.identical is True


def test_mdi_overlay_is_deterministic_and_refuses_mixed_alpha(
    null_entries: List[NullMdiEntry], tmp_path: Path
) -> None:
    """The overview figure pins like the rest and draws exactly one alpha."""
    # Given: the same series twice under two labels
    series = [(null_entries, "env A"), (null_entries, "env B")]
    # When: rendered twice
    first = render_mdi_overlay(series, str(tmp_path / "a"), stem="o")
    second = render_mdi_overlay(series, str(tmp_path / "b"), stem="o")
    # Then: byte-identical
    for left, right in zip(first, second):
        assert Path(left).read_bytes() == Path(right).read_bytes(), os.path.basename(left)
    # And: a foreign alpha in any series is refused
    foreign = dict(null_entries[0])
    foreign["alpha"] = 0.01
    with pytest.raises(ValueError, match="one alpha"):
        render_mdi_overlay(
            [(null_entries, "A"), ([foreign], "B")],  # type: ignore[list-item]
            str(tmp_path / "c"),
            stem="bad",
        )


def test_fip_overlay_is_deterministic_and_refuses_an_empty_series(
    curve: FipCurve, tmp_path: Path
) -> None:
    """The three-environment FIP figure replaced three unreadable panels; pin it."""
    # Given: two labelled curves, one with an explicit null-PI marker
    series = [(curve, 0.8, "env A"), (curve, None, "env B")]
    # When: rendered twice
    first = render_fip_overlay(series, str(tmp_path / "a"), stem="f")
    second = render_fip_overlay(series, str(tmp_path / "b"), stem="f")
    # Then: byte-identical, and an empty series is refused
    for left, right in zip(first, second):
        assert Path(left).read_bytes() == Path(right).read_bytes(), os.path.basename(left)
    with pytest.raises(ValueError, match="no series"):
        render_fip_overlay([], str(tmp_path / "d"), stem="none")


def test_png_bytes_do_not_depend_on_whether_a_pdf_was_written_first(tmp_path: Path) -> None:
    """The PNG must not change because a PDF shared the render (AGENTS.md §3.3).

    ``_save`` draws once and then pins the layout engine, because the
    constrained-layout solver used to run again between the two ``savefig``
    calls and settle a hair differently. Rendering the same format set twice
    cannot see that -- the defect is *format-order* dependence, so both renders
    carry it equally. This varies the format set instead, on the two-panel
    figure where the shift is larger than rounding.
    """
    # Given: the promotion figure, rendered with and without a PDF sharing it
    rows, stop = _promotion_rows(), _stopping(halt=2, peak=0.09)
    with_pdf = render_promotion_stopping(
        rows, stop, str(tmp_path / "both"), stem="s", formats=("pdf", "png")
    )
    png_only = render_promotion_stopping(
        rows, stop, str(tmp_path / "png"), stem="s", formats=("png",)
    )
    # When: the two PNGs are compared
    left = next(path for path in with_pdf if path.endswith(".png"))
    right = next(path for path in png_only if path.endswith(".png"))
    # Then: writing a PDF first left the PNG untouched
    assert Path(left).read_bytes() == Path(right).read_bytes()
