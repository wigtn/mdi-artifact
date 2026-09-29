"""Deterministic paper figures for the FIP and decay curves (FR-014, PRD §4.1).

``paper/figures/`` is a *generated* tree (AGENTS.md §6): the only legal way to
change it is to re-run this code through ``mdi report``. Two consecutive renders
of the same inputs must be byte-identical, so every source of drift is pinned
here:

- ``SOURCE_DATE_EPOCH`` is set to :data:`FIGURE_EPOCH` for the duration of a
  render and the PDF ``CreationDate`` is passed explicitly — no wall-clock ever
  reaches the file;
- rcParams are reset to Matplotlib's built-in defaults first, so a user's
  ``matplotlibrc`` cannot leak into the output, then overridden by
  :data:`RC_PARAMS`;
- PDF ``Creator``/``Producer`` and PNG ``Software`` are fixed strings rather
  than the Matplotlib version banner;
- inputs arrive already sorted from the analysis layer and nothing here draws a
  random number.

Colour: two categorical slots (blue, orange) plus recessive ink/grid greys,
validated for colour-vision deficiency separation and lightness on a white
(print) surface. Reference lines are grey and directly labelled, never a third
hue.

Renderer rules (ADR-015c + first-author decisions of 2026-08-02):

a. **Noise floor display** — "omega = 0.000" is never printed. When the omega^2
   point estimate is zero or its CI lower bound is zero, the floor renders as
   "below detection at this scale (<= upper CI bound)", and every decay caption
   states which axis the run measured (single-paraphrase runs measure the
   intrinsic axis only, so no floor is expected by design; it becomes
   measurable only with the procedural axis, Exp 3).
b. **MDI-vs-N monotonicity** — the isotonic (non-increasing in N) smoothing is
   a declared method: figures draw the smoothed sequence as the line, keep the
   raw estimates as light markers, and show the bootstrap CI band.
c. **ADR-001 notation** — axes and value labels use scale-normalized
   percentage points (%p) as the primary unit; raw-scale values live in the
   derived JSON. The FIP figure's MDI label carries the null-PI (primary)
   value; the FIP-crossing validation value goes into the caption.
"""

import contextlib
import datetime
import math
import os
import textwrap
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, cast

import matplotlib
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import NullFormatter, ScalarFormatter

from mdi.stats.decay import DecayFit
from mdi.stats.fip import FipCurve
from mdi.stats.mdi_table import DEFAULT_ALPHA, MdiEntry, mdi_from_curve
from mdi.stats.null_mdi import NullMdiEntry
from mdi.stats.scales import to_pp

FIGURE_EPOCH: int = 1735689600
"""2025-01-01T00:00:00Z — the pinned SOURCE_DATE_EPOCH for every rendered figure."""

DEFAULT_FORMATS: Tuple[str, ...] = ("pdf", "png")

FOOTER_BAND: float = 0.055
"""Fraction of figure height reserved for the provenance footer."""

CAPTION_LINE_BAND: float = 0.045
"""Fraction of figure height reserved per wrapped caption line."""

CAPTION_WRAP: int = 110
"""Deterministic wrap width (characters) for caption text."""

PROVENANCE_WRAP: int = 96
"""Deterministic wrap width (characters) for the provenance footer.

Unwrapped provenance overflowed the canvas: the trailing input digest was
clipped on every render before 2026-08-05.
"""

HEADLINE_FIP_ENVS: Tuple[Tuple[str, str, str], ...] = (
    ("exp1_dense", "e_250190cb1b81", "gpt-4o-mini · Summ · L5"),
    ("exp1_anchor", "e_b842cbff3261", "gpt-5.6 · Summ · L5"),
    ("exp1_alpaca", "e_dabfcd0d96e0", "4o-mini · Instr · L5"),
)
"""(experiment, env_id, legend label) triples overlaid in the paper's FIP figure.

Fixed and ordered so the render is deterministic and the manuscript caption can
name the series in the same order. Envs absent from the derived tree are
skipped, so the overlay degrades to whatever is present rather than failing.
"""

TEASER_MDI_ENVS: Tuple[Tuple[str, str, str], ...] = (
    ("exp1_dense", "e_250190cb1b81", "4o-mini · Summ · L5"),
    ("exp1_dense", "e_7268666f4d91", "4o-mini · Summ · L10"),
    ("exp1_dense", "e_200f3c74cbfa", "4o-mini · Summ · S100"),
    ("exp1_alpaca", "e_dabfcd0d96e0", "4o-mini · Instr · L5"),
    ("exp1_alpaca", "e_9dae315a7446", "4o-mini · Instr · L10"),
    ("exp1_alpaca", "e_1cfecd45841e", "4o-mini · Instr · S100"),
    ("exp1_anchor", "e_b842cbff3261", "gpt-5.6 · Summ · L5 (anchor)"),
    ("exp1_anchor_scales", "e_d26d4476fba5", "gpt-5.6 · Summ · S100 (anchor)"),
)
"""(experiment, env_id, legend label) for the overview figure's MDI-vs-N fan.

The same eight environments as the Experiment 1 lookup table. Dense-tier rows
come first and anchor rows last, because the anchor cells use fewer items and
fewer repeats: they are drawn recessively (grey, thin, no marker) and are
excluded from the spread the caption quotes, exactly as Table 1 excludes them
from the judge comparison. Listed explicitly rather than discovered so the
render is deterministic and the selection is auditable; missing envs are
skipped.
"""

SERIES_PRIMARY: str = "#2a78d6"
SERIES_SECONDARY: str = "#eb6834"
INK_PRIMARY: str = "#0b0b0b"
INK_SECONDARY: str = "#52514e"
INK_MUTED: str = "#8a8983"
GRID_COLOR: str = "#e2e1dc"

WIDE_FIGSIZE: Tuple[float, float] = (6.8, 1.6)
"""Canvas for the overlay renders, which the paper includes at ``width=\\linewidth``.

The NeurIPS text block is 5.5 in wide, so a 6.8 in canvas is placed at 0.81x
and the 9 pt text below renders near 7.3 pt -- a touch under the caption size,
which is what a figure label should be.
"""

DECAY_FIGSIZE: Tuple[float, float] = (6.8, 2.6)
"""Canvas for the decay curve, the one single-panel render the paper includes.

Authored at the same 6.8 in as the overlays so it lands on the same 0.81x
footing at ``width=\\linewidth``. On the 3.8 in default it was *enlarged* to
1.13x (the manuscript placed it at 0.78 linewidth = 4.29 in), so its labels
came out near body size and around 40 percent larger than every other figure
in the paper. The remaining 3.8 in default belongs to the per-environment
diagnostic renders, which the manuscript does not include.
"""

RC_PARAMS: Dict[str, Any] = {
    "figure.figsize": (3.8, 2.45),
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "standard",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9.0,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9.0,
    "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK_PRIMARY,
    "axes.titlecolor": INK_PRIMARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.6,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.hashsalt": "mdi",
    "path.simplify": False,
}


@contextlib.contextmanager
def deterministic_render() -> Iterator[None]:
    """Pin rcParams and ``SOURCE_DATE_EPOCH`` for the duration of a render."""
    saved_rc = dict(matplotlib.rcParams)
    saved_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    try:
        matplotlib.rcdefaults()
        matplotlib.rcParams.update(cast(Any, RC_PARAMS))
        os.environ["SOURCE_DATE_EPOCH"] = str(FIGURE_EPOCH)
        yield
    finally:
        matplotlib.rcParams.update(cast(Any, saved_rc))
        if saved_epoch is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = saved_epoch


def _metadata(fmt: str, title: str) -> Dict[str, Any]:
    """Fixed file metadata per format — no version banners, no wall-clock."""
    if fmt == "pdf":
        return {
            "Title": title,
            "Author": "",
            "Subject": "MDI estimation framework",
            "Keywords": "MDI FIP LLM-as-a-judge",
            "Creator": "mdi",
            "Producer": "mdi",
            "CreationDate": datetime.datetime.fromtimestamp(FIGURE_EPOCH, tz=datetime.timezone.utc),
        }
    if fmt == "png":
        return {"Software": "mdi"}
    return {}


def _save(fig: Figure, out_dir: str, stem: str, formats: Sequence[str], title: str) -> List[str]:
    """Save *fig* into *out_dir* for each format; returns the sorted paths written.

    The layout is solved **once** and then frozen. Constrained layout is an
    iterative solver that re-runs on every draw, so saving the same figure to
    two formats in a row let the second save start from the first save's
    geometry and settle a hair differently. On single-axes figures the
    difference vanished in rounding; on the two-panel promotion figure it moved
    a pixel and broke the byte-identical guarantee (AGENTS.md 3.3) for the PNG
    while the PDF stayed identical. Draw once, pin the result, then write.
    """
    os.makedirs(out_dir, exist_ok=True)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    written: List[str] = []
    for fmt in sorted(set(formats)):
        path = os.path.join(out_dir, f"{stem}.{fmt}")
        fig.savefig(path, format=fmt, metadata=_metadata(fmt, title))
        written.append(path)
    return sorted(written)


def _footer(fig: Figure, provenance: Optional[str], caption: Optional[str] = None) -> None:
    """Stamp a caption and a provenance line (env id, digest, run ids) under the axes.

    The caption sits above the provenance footer; both live in a reserved
    bottom band so the axes never overlap them. Caption wrapping is fixed at
    :data:`CAPTION_WRAP` characters (deterministic).
    """
    wrapped = textwrap.fill(caption, width=CAPTION_WRAP) if caption else None
    caption_lines = wrapped.count("\n") + 1 if wrapped else 0
    # Provenance is a single long identifier string; wrap it too, or it runs off
    # the right edge of the canvas (constrained layout does not manage fig.text).
    prov = textwrap.fill(provenance, width=PROVENANCE_WRAP) if provenance else None
    prov_lines = prov.count("\n") + 1 if prov else 0
    band = FOOTER_BAND * prov_lines + CAPTION_LINE_BAND * caption_lines
    if band == 0.0:
        return
    # Constrained-layout rect is (left, bottom, width, height): reserve a bottom band.
    fig.set_layout_engine("constrained", rect=(0.0, band, 1.0, 1.0 - band))
    if prov:
        fig.text(0.006, 0.012, prov, fontsize=5.5, color=INK_MUTED, ha="left", va="bottom")
    if wrapped:
        y = 0.012 + FOOTER_BAND * prov_lines
        fig.text(0.006, y, wrapped, fontsize=5.0, color=INK_SECONDARY, ha="left", va="bottom")


def _new_figure(figsize: Optional[Tuple[float, float]] = None) -> Tuple[Figure, Axes]:
    """Create a constrained-layout figure with an Agg canvas attached."""
    fig = Figure(figsize=figsize) if figsize is not None else Figure()
    FigureCanvasAgg(fig)
    fig.set_layout_engine("constrained")
    axes = fig.subplots()
    return fig, axes


def noise_floor_display(fit: DecayFit) -> str:
    """Noise-floor label under renderer rule (a) — never "omega = 0.000".

    A floor whose point estimate is zero, or whose CI lower bound touches zero,
    is *below detection at this scale*, bounded by the CI's upper edge when one
    exists; only a floor whose interval excludes zero is quoted as a value
    (in ADR-001 %p).
    """
    ci_lo = fit.get("omega_sq_ci_lo")
    below = fit["omega_sq"] <= 0.0 or (ci_lo is not None and ci_lo <= 0.0)
    if below:
        omega_ci_hi = fit.get("omega_ci_hi")
        if omega_ci_hi is None:
            return "noise floor: below detection at this scale"
        bound_pp = to_pp(omega_ci_hi, fit["scale"])
        return f"noise floor: below detection at this scale (≤ {bound_pp:.2f} %p)"
    return f"noise floor ω = {to_pp(fit['omega'], fit['scale']):.3g} %p"


GROUP_FIELD_LABELS: Dict[str, str] = {"paraphrase_id": "prompt-paraphrase"}
"""Reader-facing name per grouping axis, used in figure captions only.

A caption is read by someone holding the paper, not the record schema, so it
names the axis rather than the field the axis is stored under. Unmapped fields
fall through to their own name: a caption that says nothing useful is worse
than one that says the field name.
"""


def decay_caption(fit: DecayFit) -> str:
    """Measured-axis caption for every decay figure (renderer rule a).

    Single-paraphrase runs measure the intrinsic axis only, so the mandated
    sentence is stamped verbatim; when the procedural axis is present the
    caption states so instead (stamping "single paraphrase" on a multi-group
    run would be false — adaptation flagged for first-author review).
    """
    if fit["mean_groups_per_cell"] <= 1.0:
        return (
            "This run measures the intrinsic axis only (single paraphrase); a noise floor is "
            "not expected by design and becomes measurable only with the procedural axis (Exp 3)."
        )
    axis = GROUP_FIELD_LABELS.get(fit["group_field"], fit["group_field"])
    return (
        f"Procedural axis included: {fit['mean_groups_per_cell']:.1f} {axis} "
        "groups per cell enter the floor estimate."
    )


def promotion_stopping_title(seed: int, clears: Sequence[bool]) -> str:
    """Title for the Experiment 4 stopping figure, read off the lower panel.

    The title this replaced ("the loop drifts past its own peak") narrated the
    promotion sequence as one trajectory. It is not one: each confirm delta is
    measured against the state in force at its own decision, and on the pinned
    seed those states differ (the Experiment 4 appendix paragraph carries the
    six baselines). Counting how many decisions fall inside the band makes the
    same point from the panel below, with no ordering implied.
    """
    inside = sum(1 for clear in clears if not clear)
    return f"Seed {seed}: {inside} of {len(clears)} accepted promotions sit inside the noise band"


def render_fip_curve(
    curve: FipCurve,
    out_dir: str,
    *,
    stem: Optional[str] = None,
    alpha: float = DEFAULT_ALPHA,
    mdi_entry: Optional[MdiEntry] = None,
    null_mdi: Optional[float] = None,
    formats: Sequence[str] = DEFAULT_FORMATS,
    provenance: Optional[str] = None,
) -> List[str]:
    """Render one FIP curve: observed delta (%p) vs flip probability, CI band, MDI marker.

    Per renderer rule (c): the axis is in ADR-001 %p, the MDI label carries the
    **null-PI (primary)** value when *null_mdi* (raw score units) is given, and
    the caption carries the FIP-crossing validation read-off — taken from the
    same curve (:func:`mdi_from_curve`) unless an *mdi_entry* is supplied, so
    the figure and the lookup table can never disagree. Without *null_mdi* the
    marker falls back to the validation value, explicitly labelled as such.
    """
    entry = mdi_entry if mdi_entry is not None else mdi_from_curve(curve, alpha=alpha)
    populated = [
        b
        for b in curve["bins"]
        if b["n_draws"] > 0 and b["fip"] is not None and b["mean_delta"] is not None
    ]
    if not populated:
        raise ValueError(f"env_id={curve['env_id']}: FIP curve has no populated bin to plot")

    scale = curve["scale"]
    xs = [to_pp(b["mean_delta"] or 0.0, scale) for b in populated]
    ys = [b["fip"] or 0.0 for b in populated]
    lo = [b["ci_lo"] if b["ci_lo"] is not None else b["fip"] or 0.0 for b in populated]
    hi = [b["ci_hi"] if b["ci_hi"] is not None else b["fip"] or 0.0 for b in populated]

    validation_pp = None if entry["mdi"] is None else to_pp(entry["mdi"], scale)
    if validation_pp is None:
        validation_text = f"FIP-crossing read-off: not attained at α = {alpha:g}."
    else:
        validation_text = f"FIP-crossing read-off at α = {alpha:g}: {validation_pp:.2f} %p."
    marker_label: Optional[str]
    if null_mdi is not None:
        marker_pp: Optional[float] = to_pp(null_mdi, scale)
        marker_label = f"MDI = {marker_pp:.2f} %p (null-PI)"
        caption = validation_text
    else:
        marker_pp = validation_pp
        marker_label = (
            None if marker_pp is None else f"MDI = {marker_pp:.2f} %p (FIP-crossing, validation)"
        )
        caption = (
            "Null-PI MDI (primary) unavailable for this render; the marked value is "
            "the FIP-crossing validation read-off. " + validation_text
        )

    label = "single-run" if curve["n_repeats"] == 1 else f"N = {curve['n_repeats']}"
    title = f"FIP curve: {curve['task']} / {curve['scale']} ({label})"

    with deterministic_render():
        fig, axes = _new_figure()
        axes.fill_between(xs, lo, hi, color=SERIES_PRIMARY, alpha=0.18, linewidth=0.0)
        axes.plot(xs, ys, color=SERIES_PRIMARY, marker="o", zorder=3)
        axes.axhline(alpha, color=INK_MUTED, linestyle="--", linewidth=1.0, zorder=2)
        axes.text(
            xs[-1],
            alpha,
            f"  α = {alpha:g}",
            color=INK_SECONDARY,
            fontsize=8.5,
            va="bottom",
            ha="right",
        )
        if marker_pp is not None and marker_label is not None:
            axes.axvline(marker_pp, color=SERIES_SECONDARY, linestyle=":", linewidth=1.4, zorder=2)
            axes.annotate(
                marker_label,
                xy=(marker_pp, alpha),
                xytext=(4, 10),
                textcoords="offset points",
                color=SERIES_SECONDARY,
                fontsize=8.5,
            )
        else:
            axes.text(
                0.98,
                0.95,
                f"MDI not attained at α = {alpha:g}",
                transform=axes.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color=SERIES_SECONDARY,
            )
        axes.set_xlabel("Observed improvement Δ (%p, scale-normalized)")
        axes.set_ylabel("Reversal probability  FIP(Δ)")
        axes.set_title(title)
        axes.set_ylim(0.0, min(1.0, max(0.2, max(hi) * 1.25)))
        x_upper = max([*xs, marker_pp] if marker_pp is not None else xs)
        axes.set_xlim(0.0, x_upper * 1.08)
        axes.grid(axis="both")
        _footer(fig, provenance, caption)
        stem_name = stem or f"fip_curve_{curve['env_id']}_N{curve['n_repeats']}"
        return _save(fig, out_dir, stem_name, formats, title)


FAN_PALETTE: Tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
    "#7F7F7F",  # grey
)
"""One colour per overlaid series (Okabe-Ito, colour-vision-deficiency safe).

Encoding colour by answer scale gave three colours for eight curves, so four
series shared a hue and the fan read as a tangle. Every series now owns a
colour; the task still sets the dash pattern, which keeps the two tasks
separable in greyscale.
"""

TASK_DASH: Dict[str, str] = {"summarization": "-", "instruction_following": "--"}
"""Overlay dash pattern per task, so the two tasks separate in greyscale."""

FAN_MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
"""One marker per overlaid series — a third cue on top of colour and dash."""


def render_mdi_overlay(
    series: Sequence[Tuple[Sequence[NullMdiEntry], str]],
    out_dir: str,
    *,
    stem: str,
    formats: Sequence[str] = DEFAULT_FORMATS,
    provenance: Optional[str] = None,
) -> List[str]:
    """Overlay MDI(env, N) for several environments — the paper's overview figure.

    Each element of *series* is ``(entries, label)`` for one environment. The
    smoothed sequence is drawn (renderer rule b); the bootstrap CI band is
    deliberately omitted because eight overlapping bands are unreadable — the
    per-environment :func:`render_mdi_curve` renders keep the bands.

    Style is semantic, not a blind cycle: colour encodes the answer scale and
    the dash pattern encodes the task, so a reader can decode the fan without
    matching eight legend entries one by one.
    """
    if not series:
        raise ValueError("render_mdi_overlay: no series to plot")

    alphas = {entry["alpha"] for entries, _ in series for entry in entries}
    if len(alphas) != 1:
        raise ValueError(f"render_mdi_overlay draws one alpha, got {sorted(alphas)}")
    alpha = alphas.pop()

    title = f"MDI by environment and budget (α = {alpha:g})"
    with deterministic_render():
        fig, axes = _new_figure(WIDE_FIGSIZE)
        y_upper = 0.0
        y_lower = float("inf")
        budgets: List[float] = []
        for index, (entries, label) in enumerate(series):
            if not entries:
                continue
            ordered = sorted(entries, key=lambda entry: entry["n_repeats"])
            head = ordered[0]
            xs = [float(entry["n_repeats"]) for entry in ordered]
            ys = [entry["mdi_null_smoothed_pp"] for entry in ordered]
            recessive = label.endswith("(anchor)")
            if recessive:
                # Fewer items and fewer repeats than the dense tier, so these
                # are context, not part of the quoted spread (same rule the
                # Experiment 1 table applies to the judge comparison).
                axes.plot(
                    xs,
                    ys,
                    color=INK_MUTED,
                    linestyle=(0, (1, 2)),
                    linewidth=1.1,
                    label=label,
                    zorder=2,
                )
            else:
                axes.plot(
                    xs,
                    ys,
                    color=FAN_PALETTE[index % len(FAN_PALETTE)],
                    linestyle=TASK_DASH.get(head["task"], "-"),
                    marker=FAN_MARKERS[index % len(FAN_MARKERS)],
                    markersize=3.6,
                    label=label,
                    zorder=3,
                )
            y_upper = max(y_upper, max(ys))
            y_lower = min(y_lower, min(ys))
            budgets.extend(xs)

        axes.set_xlabel("Scoring repeats per system, N")
        axes.set_ylabel("$\\mathrm{MDI}$ (%p)")
        axes.set_title(title)
        # Tick the budgets actually swept, not the linear locator's 2/4/6/8/10.
        axes.set_xticks(sorted(set(budgets)))
        # Log y: MDI spans an order of magnitude (0.3 to 3.0 %p), so on a linear
        # axis the anchor curve stretched the range and squashed the other six
        # into a band. The paper's claim is a ratio ("2.7x spread"), which a log
        # axis shows directly as vertical offset.
        axes.set_yscale("log")
        axes.yaxis.set_major_formatter(ScalarFormatter())
        axes.yaxis.set_minor_formatter(NullFormatter())
        # Three ticks, not five: at this panel height five log labels collided.
        axes.set_yticks([0.3, 1.0, 3.0])
        axes.set_ylim(y_lower * 0.85 if y_lower > 0 else 0.1, y_upper * 1.15)
        axes.grid(axis="both")
        handles, labels = axes.get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside right center", handlelength=2.6)
        # No in-figure caption: the manuscript \caption carries the reading, and at
        # 5 pt the duplicate was unreadable anyway (first-author note 2026-08-06).
        _footer(fig, provenance)
        return _save(fig, out_dir, stem, formats, title)


def render_promotion_stopping(
    rows: Sequence[Mapping[str, Any]],
    stopping: Mapping[str, Any],
    out_dir: str,
    *,
    stem: str,
    alpha: float = DEFAULT_ALPHA,
    formats: Sequence[str] = DEFAULT_FORMATS,
    provenance: Optional[str] = None,
) -> List[str]:
    """Render one seed's promotion trace: held-out gain above, noise probability below.

    *rows* are the seed's promotion records in loop order (``promo_num``,
    ``confirm_delta``, ``reversal_prob``, ``clears_noise_band``) and *stopping*
    is the matching entry of the report's ``stopping`` list, supplying the halt
    index and the peak/final gains.

    Added 2026-08-06: this figure previously existed in ``paper/figures/`` with
    no generator anywhere in the tree, so it could not be regenerated and the
    reproducibility claim in the appendix did not hold for it (AGENTS.md 3.3,
    6). Everything drawn here now comes from ``data/derived/exp4``.
    """
    if not rows:
        raise ValueError("render_promotion_stopping: no promotion rows for this seed")

    ordered = sorted(rows, key=lambda r: int(r["promo_num"]))
    xs = [int(r["promo_num"]) for r in ordered]
    gains = [float(r["confirm_delta"]) for r in ordered]
    probs = [float(r["reversal_prob"]) for r in ordered]
    clears = [bool(r["clears_noise_band"]) for r in ordered]
    # Seeds where no promotion clears the band have no halt point (ADR-017): the
    # rule would have taken nothing, so there is no vertical to draw.
    halt_raw = stopping.get("halt_promo_num")
    halt: Optional[int] = None if halt_raw is None else int(halt_raw)
    peak_raw = stopping.get("peak_confirm_delta")
    peak: Optional[float] = None if peak_raw is None else float(peak_raw)
    final = float(stopping["final_confirm_delta"])

    title = promotion_stopping_title(int(stopping["seed"]), clears)
    with deterministic_render():
        fig = Figure(figsize=(5.5, 1.9))
        FigureCanvasAgg(fig)
        fig.set_layout_engine("constrained")
        top, bottom = fig.subplots(2, 1, sharex=True)

        for axes in (top, bottom):
            if halt is None:
                axes.axvspan(
                    min(xs) - 0.5, max(xs) + 0.5, color=SERIES_SECONDARY, alpha=0.10, lw=0.0
                )
                continue
            if halt < max(xs):
                axes.axvspan(halt + 0.5, max(xs) + 0.5, color=SERIES_SECONDARY, alpha=0.10, lw=0.0)
            axes.axvline(halt, color=INK_SECONDARY, linestyle=":", linewidth=1.2, zorder=1)

        # Markers only: each gain is a marginal against its own baseline, and
        # the six baselines differ, so a line through the markers would draw a
        # trajectory no quantity follows.
        top.plot(xs, gains, color=SERIES_PRIMARY, marker="o", linestyle="none", zorder=3)
        top.axhline(0.0, color=INK_MUTED, linewidth=0.8, zorder=1)
        if halt is not None and peak is not None:
            top.annotate(
                f"largest band-clearing {peak:+.2f}",
                xy=(halt, peak),
                xytext=(0, 9),
                textcoords="offset points",
                color=SERIES_PRIMARY,
                fontsize=6.5,
                ha="center",
            )
        else:
            top.text(
                0.02,
                0.92,
                "no accepted promotion clears the band",
                transform=top.transAxes,
                fontsize=6.5,
                color=SERIES_SECONDARY,
                va="top",
            )
        # Both labels used to narrate a path ("peak ...", "drifts to ..."). The
        # deltas are marginals against different states, so they name what they
        # mark instead: the largest band-clearing one, and the last decision
        # taken.
        top.annotate(
            f"last decision {final:+.2f}",
            xy=(xs[-1], gains[-1]),
            xytext=(-6, 12),
            textcoords="offset points",
            color=SERIES_SECONDARY,
            fontsize=6.5,
            ha="right",
        )
        top.set_ylabel("Marginal gain\nvs. own baseline")

        colours = [SERIES_PRIMARY if ok else INK_MUTED for ok in clears]
        bottom.bar(xs, probs, color=colours, width=0.62, zorder=3)
        bottom.axhline(alpha, color=SERIES_SECONDARY, linestyle="--", linewidth=1.0, zorder=2)
        # Parked in the empty upper-left of the panel: on the line itself it
        # collided with whichever bar happened to sit near alpha.
        bottom.text(
            0.015,
            0.94,
            f"- - -  noise band α = {alpha:g}",
            transform=bottom.transAxes,
            color=SERIES_SECONDARY,
            fontsize=6.5,
            va="top",
            ha="left",
        )
        for x_pos, prob in zip(xs, probs):
            # The halt marker is a vertical dotted line through this x; a
            # centred label lands on top of it, so that one steps aside.
            on_halt = halt is not None and x_pos == halt
            bottom.annotate(
                f"{prob:.2f}",
                xy=(x_pos, prob),
                xytext=(7, 3) if on_halt else (0, 3),
                textcoords="offset points",
                fontsize=6.0,
                color=INK_SECONDARY,
                ha="left" if on_halt else "center",
            )
        bottom.set_ylabel("P(noise)")
        bottom.set_xlabel("Promotion index (accept-if-better decision, in loop order)")
        bottom.set_xticks(xs)
        bottom.set_ylim(0.0, max(1.0, max(probs) * 1.2))

        # This figure is authored at the text width, so it is placed at scale
        # 1.0 and its 9pt rcParams render at body size -- larger than every
        # other figure (all of which are authored wider and scale down) and
        # tight enough to collide. Size the text here rather than globally.
        for axes in (top, bottom):
            axes.tick_params(labelsize=6.5)
            axes.xaxis.label.set_size(7.0)
            axes.yaxis.label.set_size(7.0)

        fig.suptitle(title, fontsize=7.5)
        _footer(fig, provenance)
        return _save(fig, out_dir, stem, formats, title)


OVERLAY_STYLES: Tuple[Tuple[str, str, str], ...] = (
    (SERIES_PRIMARY, "-", "o"),
    (SERIES_SECONDARY, "--", "s"),
    (INK_SECONDARY, "-.", "^"),
)
"""(colour, linestyle, marker) per overlaid series — distinguishable in greyscale."""


def render_fip_overlay(
    series: Sequence[Tuple[FipCurve, Optional[float], str]],
    out_dir: str,
    *,
    stem: str,
    alpha: float = DEFAULT_ALPHA,
    formats: Sequence[str] = DEFAULT_FORMATS,
    provenance: Optional[str] = None,
) -> List[str]:
    """Overlay several environments' FIP curves on one axes, at manuscript width.

    Each element of *series* is ``(curve, null_mdi, label)``; ``null_mdi`` is the
    primary (null-PI) MDI in raw score units for that environment, or ``None``.
    Series are drawn in the given order with distinct colour, dash pattern and
    marker (:data:`OVERLAY_STYLES`), so the figure survives greyscale printing.

    Replaces the three side-by-side single-environment panels, which the
    manuscript had to scale to ~26 percent of the text width to fit — small
    enough that no axis label was legible.
    """
    if not series:
        raise ValueError("render_fip_overlay: no series to plot")

    title = f"Reversal probability by environment (single run, α = {alpha:g})"
    with deterministic_render():
        fig, axes = _new_figure(WIDE_FIGSIZE)
        x_upper = 0.0
        y_upper = 0.0
        for index, (curve, null_mdi, label) in enumerate(series):
            populated = [
                b
                for b in curve["bins"]
                if b["n_draws"] > 0 and b["fip"] is not None and b["mean_delta"] is not None
            ]
            if not populated:
                continue
            colour, dash, marker = OVERLAY_STYLES[index % len(OVERLAY_STYLES)]
            scale = curve["scale"]
            xs = [to_pp(b["mean_delta"] or 0.0, scale) for b in populated]
            ys = [b["fip"] or 0.0 for b in populated]
            lo = [b["ci_lo"] if b["ci_lo"] is not None else b["fip"] or 0.0 for b in populated]
            hi = [b["ci_hi"] if b["ci_hi"] is not None else b["fip"] or 0.0 for b in populated]
            axes.fill_between(xs, lo, hi, color=colour, alpha=0.13, linewidth=0.0)
            legend_label = label
            if null_mdi is not None:
                legend_label = f"{label}  (MDI {to_pp(null_mdi, scale):.2f} %p)"
            axes.plot(
                xs,
                ys,
                color=colour,
                linestyle=dash,
                marker=marker,
                label=legend_label,
                zorder=3,
            )
            if null_mdi is not None:
                axes.axvline(
                    to_pp(null_mdi, scale),
                    color=colour,
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.7,
                    zorder=2,
                )
            x_upper = max(x_upper, max(xs))
            y_upper = max(y_upper, max(hi))

        axes.axhline(alpha, color=INK_MUTED, linestyle="--", linewidth=1.0, zorder=2)
        axes.text(
            0.995,
            alpha,
            f"α = {alpha:g}  ",
            transform=axes.get_yaxis_transform(),
            color=INK_SECONDARY,
            fontsize=8.0,
            va="bottom",
            ha="right",
        )
        axes.set_xlabel("Observed improvement Δ (%p, scale-normalized)")
        axes.set_ylabel("Reversal probability  FIP(Δ)")
        axes.set_title(title)
        axes.set_xlim(0.0, x_upper * 1.06 if x_upper > 0 else 1.0)
        axes.set_ylim(0.0, min(1.0, max(0.2, y_upper * 1.28)))
        axes.grid(axis="both")
        axes.legend(loc="upper right")
        _footer(fig, provenance)
        return _save(fig, out_dir, stem, formats, title)


def render_decay_curve(
    fit: DecayFit,
    out_dir: str,
    *,
    stem: Optional[str] = None,
    formats: Sequence[str] = DEFAULT_FORMATS,
    provenance: Optional[str] = None,
) -> List[str]:
    """Render the decay curve: measured SD of mean-of-N (%p) against the 1/sqrt(N) reference.

    Renderer rule (a): a floor at or CI-touching zero is labelled "below
    detection at this scale" (never "omega = 0.000") and drawn without a floor
    line; the caption states which noise axis the run measured.
    """
    points = fit["points"]
    if not points:
        raise ValueError(f"env_id={fit['env_id']}: decay fit has no points to plot")
    scale = fit["scale"]
    xs = [float(point["n"]) for point in points]
    measured = [to_pp(point["sd_mean"], scale) for point in points]
    reference = [to_pp(point["sd_theoretical"], scale) for point in points]
    floor_text = noise_floor_display(fit)
    floor_detected = not floor_text.startswith("noise floor: below detection")
    omega_pp = to_pp(fit["omega"], scale)
    title = f"Decay of the N-repeat mean: {fit['task']} / {fit['scale']}"

    with deterministic_render():
        fig, axes = _new_figure(DECAY_FIGSIZE)
        axes.plot(
            xs,
            measured,
            color=SERIES_PRIMARY,
            marker="o",
            zorder=3,
            label="measured SD of mean-of-N",
        )
        axes.plot(
            xs,
            reference,
            color=INK_MUTED,
            linestyle="--",
            linewidth=1.2,
            zorder=2,
            label="1/√N reference",
        )
        if floor_detected:
            axes.axhline(omega_pp, color=SERIES_SECONDARY, linestyle=":", linewidth=1.4, zorder=2)
            axes.text(
                xs[-1],
                omega_pp,
                f"  {floor_text}",
                color=SERIES_SECONDARY,
                fontsize=8.5,
                va="bottom",
                ha="right",
            )
        else:
            axes.text(
                0.98,
                0.86,
                floor_text,
                transform=axes.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color=SERIES_SECONDARY,
            )
        axes.set_xscale("log")
        axes.set_xticks(xs)
        axes.set_xticklabels([str(point["n"]) for point in points])
        # Suppress the log locator's minor labels ("2 x 10^0", "4 x 10^0"), which
        # collided with the explicit N ticks set above.
        axes.xaxis.set_minor_formatter(NullFormatter())
        axes.set_xlabel("Repeats averaged per evaluation, N")
        axes.set_ylabel("SD of the N-repeat mean (%p)")
        axes.set_title(title)
        upper = max(max(measured), max(reference), omega_pp if floor_detected else 0.0) * 1.15
        axes.set_ylim(0.0, upper if upper > 0 else 1.0)
        axes.legend(loc="upper right")
        _footer(fig, provenance, decay_caption(fit))
        stem_name = stem or f"decay_curve_{fit['env_id']}"
        return _save(fig, out_dir, stem_name, formats, title)


def render_mdi_curve(
    entries: Sequence[NullMdiEntry],
    out_dir: str,
    *,
    stem: Optional[str] = None,
    formats: Sequence[str] = DEFAULT_FORMATS,
    provenance: Optional[str] = None,
) -> List[str]:
    """Render MDI(N) for one (env, alpha): smoothed line, raw light markers, CI band.

    Renderer rule (b): the isotonic non-increasing smoothing is a *declared*
    method — the raw estimates stay visible as light markers and the derived
    JSON keeps both sequences; the band is the seeded cluster bootstrap CI on
    the raw MDI estimate.
    """
    if not entries:
        raise ValueError("render_mdi_curve needs at least one null-MDI entry")
    env_ids = {entry["env_id"] for entry in entries}
    alphas = {entry["alpha"] for entry in entries}
    if len(env_ids) != 1 or len(alphas) != 1:
        raise ValueError(
            f"render_mdi_curve draws one (env, alpha) series, got envs={sorted(env_ids)} "
            f"alphas={sorted(alphas)}"
        )
    ordered = sorted(entries, key=lambda entry: entry["n_repeats"])
    head = ordered[0]
    alpha = head["alpha"]
    xs = [float(entry["n_repeats"]) for entry in ordered]
    raw = [entry["mdi_null_pp"] for entry in ordered]
    smoothed = [entry["mdi_null_smoothed_pp"] for entry in ordered]
    ci_lo = [entry["ci_lo_pp"] for entry in ordered]
    ci_hi = [entry["ci_hi_pp"] for entry in ordered]
    has_band = all(lo is not None for lo in ci_lo) and all(hi is not None for hi in ci_hi)
    title = f"MDI vs N: {head['task']} / {head['scale']} (α = {alpha:g}, null-PI)"
    caption = (
        "Isotonic non-increasing smoothing over N is a declared method; raw "
        "estimates remain as light markers and in derived JSON."
    )
    if has_band:
        caption += " Band: seeded cluster bootstrap CI over systems."
    else:
        caption += " CI band undefined (single-system cluster)."

    with deterministic_render():
        fig, axes = _new_figure()
        if has_band:
            axes.fill_between(
                xs,
                cast(List[float], ci_lo),
                cast(List[float], ci_hi),
                color=SERIES_PRIMARY,
                alpha=0.18,
                linewidth=0.0,
            )
        axes.plot(
            xs,
            raw,
            color=SERIES_PRIMARY,
            marker="o",
            linestyle="none",
            alpha=0.35,
            zorder=3,
            label="raw MDI_null(N)",
        )
        axes.plot(
            xs,
            smoothed,
            color=SERIES_PRIMARY,
            marker="D",
            markersize=3.2,
            zorder=4,
            label="isotonic (non-increasing) MDI_null(N)",
        )
        axes.set_xscale("log")
        axes.set_xticks(xs)
        axes.set_xticklabels([str(entry["n_repeats"]) for entry in ordered])
        # A log axis that spans less than a decade grows minor tick labels; the
        # explicit N ticks above are the only labels this figure should carry.
        axes.xaxis.set_minor_formatter(NullFormatter())
        axes.set_xlabel("Repeats averaged per evaluation, N")
        axes.set_ylabel("MDI (%p, null-PI half-width)")
        axes.set_title(title)
        upper_values: List[float] = [*raw, *smoothed]
        if has_band:
            upper_values.extend(value for value in ci_hi if value is not None)
        upper = max(upper_values) * 1.15
        axes.set_ylim(0.0, upper if upper > 0 else 1.0)
        axes.legend(loc="upper right")
        _footer(fig, provenance, caption)
        stem_name = stem or f"mdi_null_curve_{head['env_id']}"
        return _save(fig, out_dir, stem_name, formats, title)


def sigma_reference(fit: DecayFit) -> float:
    """Convenience: the fitted intrinsic sigma of an environment (sqrt of the slope)."""
    return math.sqrt(max(fit["sigma_sq"], 0.0))
