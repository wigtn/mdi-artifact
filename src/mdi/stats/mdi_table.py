"""MDI lookup tables (FR-009): the null-PI definition, validated by the FIP read-off.

ADR-015 splits the table into two paths:

- **Primary (ADR-015a)** — ``MDI_null(env, N, alpha)``, the half-width of the
  central ``(1 - alpha)`` prediction interval of the null distribution
  (:mod:`mdi.stats.null_mdi`). This is the canonical definition; headline
  numbers, figure labels and the primary table column come from it.
- **Validation (ADR-015b)** — the FIP-crossing read-off implemented here:
  the smallest observed improvement whose flip probability sits at or below
  ``alpha``, read off the measured FIP curve. Both paths are computed on the
  same store and emitted side by side with a ``consistency`` block
  (``{mdi_null, mdi_fip_crossing, ratio, abs_diff}`` per ``(env, N, alpha)``);
  divergence is reported as-is, never hidden.

The output is a lookup table over ``(task x scale x N)`` with alpha in
``{0.10, 0.05, 0.01}`` (FR-013 appendix sensitivity), each row carrying the
environment it was measured in, the sigma of that environment, and MDI expressed
in sigma units so environments remain comparable.

Methodology defaults flagged for first-author review (not fixed by an Accepted
ADR):

1. :data:`DEFAULT_ALPHA` = 0.05 as the headline error level (0.10 / 0.01 also
   emitted). PRD §5.2/FR-009 fixes the 95% headline for prediction intervals;
   applying it to the FIP read-off is the natural reading, not a decided one.
2. **Read-off rule** ``interpolate``: bins are represented by their mean observed
   delta, and MDI is linearly interpolated between the last bin above alpha and
   the first bin at or below it. ``bin_upper`` (report the bin's upper edge) is
   the conservative alternative.
3. A bin needs :data:`DEFAULT_MIN_BIN_DRAWS` draws to be trusted for the
   read-off; sparser bins are ignored.
4. The crossing must be **sustained**: MDI is the smallest delta from which every
   higher bin also stays at or below alpha, so a single noisy dip cannot set it.
5. The consistency ``ratio`` is oriented validation-over-primary
   (``mdi_fip_crossing / mdi_null``) and left undefined when either side is
   missing or the primary is zero.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

from mdi.stats import null_mdi as null_mod
from mdi.stats import records as rec
from mdi.stats.bootstrap import DEFAULT_CI_LEVEL, DEFAULT_N_RESAMPLES, DEFAULT_SEED
from mdi.stats.fip import (
    DEFAULT_DRAWS_PER_PAIR,
    DEFAULT_SIGMA_BIN_MULTIPLES,
    FipCurve,
    RepeatSplit,
    estimate_fip,
)
from mdi.stats.null_mdi import NullMdiEntry
from mdi.stats.scales import from_pp, maybe_pp
from mdi.store import ScoreRecord

DEFAULT_ALPHA: float = 0.05
DEFAULT_ALPHAS: Tuple[float, ...] = (0.10, 0.05, 0.01)
DEFAULT_MIN_BIN_DRAWS: int = 30
INTERPOLATE: str = "interpolate"
BIN_UPPER: str = "bin_upper"
DEFAULT_READOFF: str = INTERPOLATE

#: Default target improvements (%p) for the reverse "how many repeats?" table.
#: These are the practitioner's input, not an environment property — a fixed
#: menu is fine precisely because MDI is env-conditional (the point of the
#: reverse table is that the SAME target needs different N in different envs).
DEFAULT_TARGETS_PP: Tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)


class MdiEntry(TypedDict):
    """One MDI lookup-table row."""

    env_id: str
    task: str
    scale: str
    n_repeats: int
    alpha: float
    mdi: Optional[float]
    mdi_in_sigma: Optional[float]
    attained: bool
    below_measured_range: bool
    readoff: str
    sigma: float
    n_draws: int
    n_bins_used: int


class ConsistencyEntry(TypedDict):
    """ADR-015b consistency check between the primary and validation MDI paths."""

    env_id: str
    task: str
    scale: str
    n_repeats: int
    alpha: float
    mdi_null: Optional[float]
    mdi_fip_crossing: Optional[float]
    ratio: Optional[float]
    abs_diff: Optional[float]
    mdi_null_pp: Optional[float]
    mdi_fip_crossing_pp: Optional[float]
    abs_diff_pp: Optional[float]


class ReverseEntry(TypedDict):
    """Practitioner-facing inverse of the MDI table (protocol artifact).

    Answers "to detect a target improvement, how many repeats do I need?" —
    the smallest sweep budget ``N`` whose ``MDI_null(env, N, alpha)`` is at or
    below the target. ``recommended_n`` is ``None`` when even the largest N in
    the sweep cannot resolve the target (the target is below the environment's
    reachable resolution at this budget).
    """

    env_id: str
    task: str
    scale: str
    alpha: float
    #: Detection probability the recommendation is read at. ``0.5`` is the
    #: alpha-level column (a true gain of exactly MDI clears it about half the
    #: time); ``power`` rows answer the question a practitioner actually asks
    #: (ADR-024a).
    power: float
    target_pp: float
    target_raw: float
    recommended_n: Optional[int]
    mdi_at_recommended_pp: Optional[float]
    reachable: bool


class MdiTable(TypedDict):
    """Full FR-009 payload: primary (null-PI) and validation (FIP-crossing) paths."""

    entries: List[MdiEntry]
    null_entries: List[NullMdiEntry]
    reverse_entries: List[ReverseEntry]
    #: The same inversion read at :data:`mdi.stats.null_mdi.DEFAULT_POWER`
    #: instead of the alpha level (ADR-024a). Kept beside the alpha-level table
    #: rather than replacing it: the two answer different questions and the gap
    #: between them is itself reportable.
    reverse_entries_at_power: List[ReverseEntry]
    power: float
    target_pp: List[float]
    consistency: List[ConsistencyEntry]
    alphas: List[float]
    readoff: str
    min_bin_draws: int
    draws_per_system: int
    primary: str
    validation: str


def _usable_points(
    curve: FipCurve,
    min_bin_draws: int,
) -> List[Tuple[float, float, Optional[float]]]:
    """(delta, fip, bin_upper_edge) for bins with enough draws, sorted by delta."""
    points: List[Tuple[float, float, Optional[float]]] = []
    for entry in curve["bins"]:
        if entry["n_draws"] < min_bin_draws:
            continue
        if entry["fip"] is None or entry["mean_delta"] is None:
            continue
        points.append((entry["mean_delta"], entry["fip"], entry["hi"]))
    points.sort(key=lambda point: point[0])
    return points


def mdi_from_curve(
    curve: FipCurve,
    *,
    alpha: float = DEFAULT_ALPHA,
    readoff: str = DEFAULT_READOFF,
    min_bin_draws: int = DEFAULT_MIN_BIN_DRAWS,
) -> MdiEntry:
    """Read ``MDI(env, N, alpha)`` off a measured FIP curve.

    Returns an entry with ``attained=False`` and ``mdi=None`` when no measured
    delta bin reaches *alpha* — the honest outcome for an environment whose
    resolution is coarser than the deltas that were probed.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if readoff not in (INTERPOLATE, BIN_UPPER):
        raise ValueError(f"unknown readoff rule: {readoff!r}")

    points = _usable_points(curve, min_bin_draws)
    sigma = curve["meta"]["sigma"]
    base = MdiEntry(
        env_id=curve["env_id"],
        task=curve["task"],
        scale=curve["scale"],
        n_repeats=curve["n_repeats"],
        alpha=alpha,
        mdi=None,
        mdi_in_sigma=None,
        attained=False,
        below_measured_range=False,
        readoff=readoff,
        sigma=sigma,
        n_draws=curve["meta"]["n_draws"],
        n_bins_used=len(points),
    )
    if not points:
        return base

    crossing: Optional[int] = None
    for index in range(len(points)):
        if all(point[1] <= alpha for point in points[index:]):
            crossing = index
            break
    if crossing is None:
        return base

    delta, _fip, upper = points[crossing]
    if crossing == 0:
        base["mdi"] = delta
        base["attained"] = True
        base["below_measured_range"] = True
    elif readoff == BIN_UPPER:
        base["mdi"] = upper if upper is not None else delta
        base["attained"] = True
    else:
        prev_delta, prev_fip, _prev_upper = points[crossing - 1]
        span = prev_fip - _fip
        if span <= 0:
            base["mdi"] = delta
        else:
            weight = (prev_fip - alpha) / span
            base["mdi"] = prev_delta + weight * (delta - prev_delta)
        base["attained"] = True
    if base["mdi"] is not None and sigma > 0:
        base["mdi_in_sigma"] = base["mdi"] / sigma
    return base


def mdi_entries_from_curves(
    curves: Sequence[FipCurve],
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    readoff: str = DEFAULT_READOFF,
    min_bin_draws: int = DEFAULT_MIN_BIN_DRAWS,
) -> List[MdiEntry]:
    """Read off every (curve x alpha) combination, sorted by task/scale/env/N/alpha."""
    entries = [
        mdi_from_curve(curve, alpha=alpha, readoff=readoff, min_bin_draws=min_bin_draws)
        for curve in curves
        for alpha in sorted(set(alphas), reverse=True)
    ]
    entries.sort(
        key=lambda entry: (
            entry["task"],
            entry["scale"],
            entry["env_id"],
            entry["n_repeats"],
            -entry["alpha"],
        )
    )
    return entries


def consistency_entries(
    null_entries: Sequence[NullMdiEntry],
    fip_entries: Sequence[MdiEntry],
) -> List[ConsistencyEntry]:
    """ADR-015b block: primary vs validation per ``(env, N, alpha)``, union of both paths.

    A key present on one side only still gets a row (the missing side is
    ``None``) — the check reports, it never filters.
    """
    null_index: Dict[Tuple[str, int, float], NullMdiEntry] = {
        (entry["env_id"], entry["n_repeats"], entry["alpha"]): entry for entry in null_entries
    }
    fip_index: Dict[Tuple[str, int, float], MdiEntry] = {
        (entry["env_id"], entry["n_repeats"], entry["alpha"]): entry for entry in fip_entries
    }
    out: List[ConsistencyEntry] = []
    for key in set(null_index) | set(fip_index):
        null_entry = null_index.get(key)
        fip_entry = fip_index.get(key)
        head = null_entry if null_entry is not None else fip_entry
        assert head is not None  # the key came from one of the two indices
        scale = head["scale"]
        mdi_null = None if null_entry is None else null_entry["mdi_null"]
        mdi_fip = None if fip_entry is None else fip_entry["mdi"]
        ratio: Optional[float] = None
        abs_diff: Optional[float] = None
        if mdi_null is not None and mdi_fip is not None:
            abs_diff = abs(mdi_null - mdi_fip)
            if mdi_null != 0.0:
                ratio = mdi_fip / mdi_null
        out.append(
            ConsistencyEntry(
                env_id=head["env_id"],
                task=head["task"],
                scale=scale,
                n_repeats=head["n_repeats"],
                alpha=head["alpha"],
                mdi_null=mdi_null,
                mdi_fip_crossing=mdi_fip,
                ratio=ratio,
                abs_diff=abs_diff,
                mdi_null_pp=maybe_pp(mdi_null, scale),
                mdi_fip_crossing_pp=maybe_pp(mdi_fip, scale),
                abs_diff_pp=maybe_pp(abs_diff, scale),
            )
        )
    out.sort(
        key=lambda entry: (
            entry["task"],
            entry["scale"],
            entry["env_id"],
            entry["n_repeats"],
            -entry["alpha"],
        )
    )
    return out


def reverse_entries_from_null(
    null_entries: Sequence[NullMdiEntry],
    *,
    targets_pp: Sequence[float] = DEFAULT_TARGETS_PP,
    alpha: float = DEFAULT_ALPHA,
    at_power: bool = False,
) -> List[ReverseEntry]:
    """Invert the null-PI table: the smallest N that resolves each target improvement.

    For each environment and each target improvement (in %p), scan the sweep in
    increasing N and return the first budget whose smoothed MDI is at or below
    the target. The smoothed (isotonic, non-increasing in N) value is used so a
    noisy dip cannot recommend a smaller N than a larger budget would need.
    Uses only ``null_entries`` — a pure transform, no resampling.

    With ``at_power`` the scan reads the power column instead of the alpha-level
    one (ADR-024a). The distinction is load-bearing rather than cosmetic: at the
    alpha-level value a true gain of exactly the target is detected about half
    the time, so an ``at_power=False`` row answers "what can be told apart from
    noise" while an ``at_power=True`` row answers "what will I actually catch".
    Budgets whose power column is undefined are skipped, not extrapolated over.
    """
    by_env: Dict[str, List[NullMdiEntry]] = {}
    for entry in null_entries:
        if abs(entry["alpha"] - alpha) > 1e-12:
            continue
        by_env.setdefault(entry["env_id"], []).append(entry)

    out: List[ReverseEntry] = []
    for env_id in sorted(by_env):
        rows = sorted(by_env[env_id], key=lambda item: item["n_repeats"])
        if not rows:
            continue
        scale = rows[0]["scale"]
        task = rows[0]["task"]
        power = rows[0]["power"] if at_power else 0.5
        for target_pp in sorted(set(targets_pp)):
            recommended_n: Optional[int] = None
            mdi_at_pp: Optional[float] = None
            for row in rows:
                value = row["mdi_power_smoothed_pp"] if at_power else row["mdi_null_smoothed_pp"]
                if value is None:
                    continue
                if value <= target_pp:
                    recommended_n = row["n_repeats"]
                    mdi_at_pp = value
                    break
            out.append(
                ReverseEntry(
                    env_id=env_id,
                    task=task,
                    scale=scale,
                    alpha=alpha,
                    power=power,
                    target_pp=target_pp,
                    target_raw=from_pp(target_pp, scale),
                    recommended_n=recommended_n,
                    mdi_at_recommended_pp=mdi_at_pp,
                    reachable=recommended_n is not None,
                )
            )
    return out


def build_mdi_table(
    all_records: Iterable[ScoreRecord],
    *,
    split: RepeatSplit,
    env_ids: Optional[Sequence[str]] = None,
    sweep: Sequence[int] = (1, 3, 5, 8, 10, 20),
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    seed: int = DEFAULT_SEED,
    readoff: str = DEFAULT_READOFF,
    min_bin_draws: int = DEFAULT_MIN_BIN_DRAWS,
    draws_per_pair: int = DEFAULT_DRAWS_PER_PAIR,
    draws_per_system: int = null_mod.DEFAULT_DRAWS_PER_SYSTEM,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    sigma_bin_multiples: Sequence[float] = DEFAULT_SIGMA_BIN_MULTIPLES,
    targets_pp: Sequence[float] = DEFAULT_TARGETS_PP,
    power: float = null_mod.DEFAULT_POWER,
) -> MdiTable:
    """Build the (task x scale x N) MDI lookup table — both ADR-015 paths.

    The primary ``null_entries`` come from :func:`mdi.stats.null_mdi.null_mdi_entries`
    (the sweep is truncated to what the estimation repeats support); the
    validation ``entries`` are read off one FIP curve per (env, N) at every
    alpha, so the appendix sensitivity rows and the headline row come from
    identical draws. ``consistency`` compares the two per ``(env, N, alpha)``.
    """
    ordered = rec.sorted_records(all_records)
    targets = sorted(env_ids) if env_ids is not None else rec.env_ids(ordered)
    curves: List[FipCurve] = []
    for env_index, env_id in enumerate(targets):
        for n_index, n_repeats in enumerate(sorted(set(sweep))):
            curves.append(
                estimate_fip(
                    ordered,
                    env_id,
                    split=split,
                    n_repeats=n_repeats,
                    pairs=pairs,
                    seed=seed + 1009 * env_index + 17 * n_index,
                    sigma_bin_multiples=sigma_bin_multiples,
                    draws_per_pair=draws_per_pair,
                    n_resamples=n_resamples,
                    ci_level=ci_level,
                )
            )
    entries = mdi_entries_from_curves(
        curves, alphas=alphas, readoff=readoff, min_bin_draws=min_bin_draws
    )
    null_entries = null_mod.null_mdi_entries(
        ordered,
        split=split,
        env_ids=targets,
        sweep=sweep,
        alphas=alphas,
        seed=seed,
        draws_per_system=draws_per_system,
        n_resamples=n_resamples,
        ci_level=ci_level,
        power=power,
    )
    return MdiTable(
        entries=entries,
        null_entries=null_entries,
        reverse_entries=reverse_entries_from_null(
            null_entries, targets_pp=targets_pp, alpha=DEFAULT_ALPHA
        ),
        reverse_entries_at_power=reverse_entries_from_null(
            null_entries, targets_pp=targets_pp, alpha=DEFAULT_ALPHA, at_power=True
        ),
        power=power,
        target_pp=sorted(set(targets_pp)),
        consistency=consistency_entries(null_entries, entries),
        alphas=sorted(set(alphas), reverse=True),
        readoff=readoff,
        min_bin_draws=min_bin_draws,
        draws_per_system=draws_per_system,
        primary="null_pi_half_width",
        validation="fip_crossing",
    )
