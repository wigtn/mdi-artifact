"""Canonical MDI (ADR-015a): half-width of the null distribution's central prediction interval.

ADR-015 makes the **null-prediction-interval** definition the primary MDI path:
``MDI(env, N, alpha)`` is the half-width of the central ``(1 - alpha)``
prediction interval of the *null distribution* — the distribution of mean score
differences between two independent scorings of the SAME system set, whose true
skill difference is zero by construction. The FIP-crossing read-off
(:func:`mdi.stats.mdi_table.mdi_from_curve`) remains as the validation path
(ADR-015b); both are emitted side by side with a consistency block, and any
divergence is reported, never hidden.

Construction, per ``(env, N)`` — entirely from measured repeats, sharing the
FIP draw machinery (:func:`mdi.stats.fip.split_pools`,
:func:`mdi.stats.fip.draw_mean`) so the two paths rest on identical sampling
assumptions:

1. Take the estimation half of the ADR-007 repeat split.
2. For each system and each draw: per item, split the cell's estimation repeats
   into two **disjoint** pools (a fresh random split per item per draw), draw an
   ``N``-repeat average from each pool (with replacement inside a pool),
   difference them, and average over the item set. Both pools come from the
   same cell, so the true difference is zero by construction.
3. Pool the draws over systems; ``MDI_null(env, N, alpha)`` is the half-width
   ``(q_{1-alpha/2} - q_{alpha/2}) / 2`` of that pooled sample.
4. A seeded cluster bootstrap over systems puts a confidence interval on the
   MDI estimate itself.

**Operationalization flagged for first-author review**: the first-author text
defines MDI as the null interval's width; because improvement claims are
directional, this module operationalizes it as the **half-width** of the
central interval. Both the half-width and the ``(1 - alpha)`` quantile of
``|delta_null|`` are recorded in derived JSON; labels use the half-width.

Methodology defaults flagged for first-author review (not fixed by ADR-015):

1. :data:`DEFAULT_DRAWS_PER_SYSTEM` null draws per ``(system, N)``.
2. **Sweep truncation**: an ``N`` is kept only when *every* cell holds at least
   ``N`` estimation repeats. With-replacement draws could simulate larger
   budgets, but reporting them would overstate what the data measured — and a
   ``max``-over-cells rule would let a single outlier cell open a whole budget
   column that the remaining cells can only simulate (ADR-018).
3. **Systems as the bootstrap cluster** (FIP clusters on system pairs); one
   system is degenerate and yields an undefined CI, never a zero-width one
   (:data:`mdi.stats.bootstrap.MIN_CLUSTERS_FOR_CI`).
4. Isotonic **non-increasing-in-N smoothing as a declared method** (ADR-015c):
   raw and smoothed values are both kept in derived JSON; figures draw the
   smoothed sequence as the line and the raw values as light markers.
"""

import bisect
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

from mdi.stats import records as rec
from mdi.stats import variance as var
from mdi.stats.bootstrap import (
    DEFAULT_CI_LEVEL,
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    BootstrapMeta,
    bootstrap_vector,
    make_rng,
    percentile,
)
from mdi.stats.fip import RepeatSplit, draw_mean, split_pools, validate_split
from mdi.stats.scales import maybe_pp, to_pp
from mdi.store import ScoreRecord

DEFAULT_DRAWS_PER_SYSTEM: int = 400
DEFAULT_SWEEP: Tuple[int, ...] = (1, 3, 5, 8, 10, 20)
#: ADR-024a: the target power the ``mdi_80`` column is read at. ``MDI`` without a
#: power subscript is the alpha-level half-width (power ~ 0.5); a team wanting a
#: stated chance of catching a true gain of that size needs the larger value.
DEFAULT_POWER: float = 0.80
#: Power-grid resolution, as a fraction of the alpha-level threshold, and its
#: upper bound in the same unit. Fixed here so the read-off is deterministic.
POWER_GRID_STEP: float = 0.005
POWER_GRID_MAX: float = 4.0
#: Kept equal to :data:`mdi.stats.mdi_table.DEFAULT_ALPHAS` (duplicated here to
#: avoid an import cycle — mdi_table imports this module).
DEFAULT_ALPHAS: Tuple[float, ...] = (0.10, 0.05, 0.01)
#: Deterministic offset separating the null-path RNG stream from the FIP
#: curves, which use the same ``seed + 1009*env + 17*n`` schedule.
NULL_SEED_OFFSET: int = 104729


class NullMdiEntry(TypedDict):
    """One primary-path MDI value: ``(env, N, alpha)`` under the null-PI definition.

    ``*_pp`` fields carry the ADR-001 primary notation (scale-normalized
    percentage points); raw-scale values stay alongside them.
    """

    env_id: str
    task: str
    scale: str
    n_repeats: int
    alpha: float
    mdi_null: float
    mdi_null_pp: float
    mdi_null_smoothed: float
    mdi_null_smoothed_pp: float
    #: ADR-024a: the smallest true gain detected with probability ``power``,
    #: read from the null draws under a location shift. ``None`` when the grid
    #: (:data:`POWER_GRID_MAX` x the alpha-level value) does not reach it.
    power: float
    mdi_power: Optional[float]
    mdi_power_pp: Optional[float]
    mdi_power_smoothed: Optional[float]
    mdi_power_smoothed_pp: Optional[float]
    ratio_power: Optional[float]
    #: ADR-025a: the split-half term this estimator adds on top of the
    #: infinite-pool estimand, and the value with it divided out.
    pool_term: float
    pool_inflation: float
    mdi_null_corrected: float
    mdi_null_corrected_pp: float
    abs_delta_quantile: float
    abs_delta_quantile_pp: float
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    ci_lo_pp: Optional[float]
    ci_hi_pp: Optional[float]
    mdi_null_in_sigma: Optional[float]
    sigma: Optional[float]
    n_draws: int
    n_systems: int
    n_items: int
    bootstrap: BootstrapMeta


class _SystemDraws(TypedDict):
    """Bootstrap cluster: every null draw belonging to one system."""

    system_id: str
    draws: List[float]


def isotonic_non_increasing(values: Sequence[float]) -> List[float]:
    """Non-increasing isotonic regression of *values* (PAVA, equal weights).

    The declared smoothing of ADR-015c: it never invents a trend, it only pools
    adjacent violators of monotonicity into their mean, and the raw sequence is
    always kept next to it.
    """
    means: List[float] = []
    counts: List[int] = []
    for value in values:
        means.append(float(value))
        counts.append(1)
        while len(means) >= 2 and means[-2] < means[-1]:
            pooled = means[-2] * counts[-2] + means[-1] * counts[-1]
            count = counts[-2] + counts[-1]
            means[-2:] = [pooled / count]
            counts[-2:] = [count]
    out: List[float] = []
    for mean, count in zip(means, counts):
        out.extend([mean] * count)
    return out


def half_width(draws: Sequence[float], alpha: float, *, assume_sorted: bool = False) -> float:
    """Half-width of the central ``(1 - alpha)`` interval of *draws* (type-7 quantiles)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    lo = percentile(draws, alpha / 2.0, assume_sorted=assume_sorted)
    hi = percentile(draws, 1.0 - alpha / 2.0, assume_sorted=assume_sorted)
    return (hi - lo) / 2.0


def mdi_at_power(
    draws: Sequence[float],
    threshold: float,
    power: float,
    *,
    grid_step: float = POWER_GRID_STEP,
    grid_max: float = POWER_GRID_MAX,
    assume_sorted: bool = False,
) -> Optional[float]:
    """Smallest true gain detected with probability *power* (ADR-024a).

    The null draws describe how an observed gain scatters when the true gain is
    zero. Assuming a true gain of ``delta`` only *shifts* that distribution --- a
    weaker assumption than normality, which is why this is read off the draws
    rather than computed as ``1 + z_{1-beta}/z_{1-alpha/2}`` --- the detection
    probability is the share of shifted draws landing above *threshold*. The
    return value is the smallest grid point reaching *power*, or ``None`` when
    ``grid_max * threshold`` does not reach it (never extrapolate: ADR-018).

    At ``power = 0.5`` and a symmetric null this returns *threshold* itself, so
    the existing alpha-level column is the ``power = 0.5`` member of the family.
    """
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    if threshold <= 0.0:
        return None
    ordered = list(draws) if assume_sorted else sorted(draws)
    total = len(ordered)
    if total == 0:
        return None
    steps = int(round(grid_max / grid_step))
    for step in range(steps + 1):
        delta = threshold * step * grid_step
        # P(delta_null + delta > threshold) = share of draws above threshold - delta
        detected = total - bisect.bisect_right(ordered, threshold - delta)
        if detected / total >= power:
            return delta
    return None


def pool_inflation(pool_term: float, n_repeats: int) -> float:
    """How much the split-half estimator exceeds the infinite-pool estimand (ADR-025a).

    One null draw differences two ``N``-repeat means taken from disjoint pools
    ``A``, ``B`` of a cell's ``r`` estimation repeats. Writing
    ``P = 1/|A| + 1/|B|``, the per-item variance is
    ``sigma^2 * [(2 - P)/N + P]`` --- the within-pool term loses ``P/N`` because a
    pool of size ``a`` has population variance ``sigma^2 (a-1)/a``, and the pools'
    own means differ by ``sigma^2 P``. The estimand this project reports a
    threshold *for* is the infinite-pool ``sigma^2 * 2/N``, so the ratio of
    standard deviations is the returned factor.

    Two consequences worth stating: at ``N = 1`` the two terms cancel exactly
    (one draw from each pool is just two draws from the ``r`` repeats), so there
    is no inflation at all; and the factor grows without bound in ``N``, which is
    why a tier with few repeats cannot carry a large-``N`` column.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    if pool_term <= 0.0:
        return 1.0
    ideal = 2.0 / n_repeats
    actual = (2.0 - pool_term) / n_repeats + pool_term
    if actual <= 0.0:
        return 1.0
    return math.sqrt(actual / ideal)


def _system_null_draws(
    cells: Dict[rec.CellKey, Dict[int, float]],
    env_id: str,
    system_id: str,
    items: Sequence[str],
    estimation: Sequence[int],
    n_repeats: int,
    draws_per_system: int,
    rng: random.Random,
) -> Tuple[List[float], float]:
    """Null delta draws for one system, and the mean split-half term they carry.

    The second return value is ``mean(1/|A| + 1/|B|)`` over every (item, draw)
    split --- the exact quantity :func:`pool_inflation` needs, accumulated rather
    than assumed so that cells holding different numbers of repeats are handled
    without a uniform-``r`` approximation.
    """
    out: List[float] = []
    pool_terms: List[float] = []
    for _ in range(draws_per_system):
        delta = 0.0
        for item_id in items:
            values = cells[rec.CellKey(env_id, item_id, system_id)]
            pool_a, pool_b = split_pools(values, estimation, rng)
            delta += draw_mean(pool_a, n_repeats, rng) - draw_mean(pool_b, n_repeats, rng)
            pool_terms.append(1.0 / len(pool_a) + 1.0 / len(pool_b))
        out.append(delta / len(items))
    return out, math.fsum(pool_terms) / len(pool_terms)


def estimate_null_mdi(
    all_records: Iterable[ScoreRecord],
    env_id: str,
    *,
    split: RepeatSplit,
    n_repeats: int,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    seed: int = DEFAULT_SEED,
    draws_per_system: int = DEFAULT_DRAWS_PER_SYSTEM,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    power: float = DEFAULT_POWER,
) -> List[NullMdiEntry]:
    """``MDI_null(env, N, alpha)`` for one ``(env, N)``, one entry per *alpha*.

    Smoothed fields are initialized to the raw value; the N-sweep builder
    (:func:`null_mdi_entries`) overwrites them with the isotonic sequence.
    """
    validate_split(split)
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    env_records = rec.filter_records(all_records, env_id=env_id)
    if not env_records:
        raise ValueError(f"no records for env_id={env_id!r}")
    rec.check_schema(env_records)
    meta = {m["env_id"]: m for m in rec.env_metadata(env_records)}[env_id]

    estimation = list(split["estimation"])
    cells = rec.group_by_cell(env_records, repeats=estimation)
    sigma = var.pooled_sigma([rec.cell_values(cell) for cell in cells.values()])

    rng = make_rng(seed)
    clusters: List[_SystemDraws] = []
    item_counts: List[int] = []
    pool_terms: List[float] = []
    for system_id in rec.system_ids(env_records):
        items = sorted(
            key.item_id for key in cells if key.system_id == system_id and len(cells[key]) >= 2
        )
        if not items:
            continue
        draws, pool_term = _system_null_draws(
            cells, env_id, system_id, items, estimation, n_repeats, draws_per_system, rng
        )
        item_counts.append(len(items))
        pool_terms.append(pool_term)
        clusters.append(_SystemDraws(system_id=system_id, draws=draws))

    if not clusters:
        raise ValueError(
            f"env_id={env_id!r}: no system has an item with >= 2 estimation repeats — "
            "the null distribution needs two disjoint scorings of the same cell"
        )

    ordered_alphas = sorted(set(alphas), reverse=True)

    def statistic(sample: Sequence[_SystemDraws]) -> Sequence[Optional[float]]:
        pooled = sorted(value for cluster in sample for value in cluster["draws"])
        return [half_width(pooled, alpha, assume_sorted=True) for alpha in ordered_alphas]

    boot = bootstrap_vector(
        clusters, statistic, seed=seed, n_resamples=n_resamples, ci_level=ci_level
    )

    pooled = sorted(value for cluster in clusters for value in cluster["draws"])
    abs_pooled = sorted(abs(value) for value in pooled)
    pool_term = math.fsum(pool_terms) / len(pool_terms)
    inflation = pool_inflation(pool_term, n_repeats)
    scale = meta["scale"]
    entries: List[NullMdiEntry] = []
    for index, alpha in enumerate(ordered_alphas):
        value = half_width(pooled, alpha, assume_sorted=True)
        abs_quantile = percentile(abs_pooled, 1.0 - alpha, assume_sorted=True)
        ci_lo = boot[index]["ci_lo"]
        ci_hi = boot[index]["ci_hi"]
        at_power = mdi_at_power(pooled, value, power, assume_sorted=True)
        corrected = value / inflation
        entries.append(
            NullMdiEntry(
                env_id=env_id,
                task=meta["task"],
                scale=scale,
                n_repeats=n_repeats,
                alpha=alpha,
                mdi_null=value,
                mdi_null_pp=to_pp(value, scale),
                mdi_null_smoothed=value,
                mdi_null_smoothed_pp=to_pp(value, scale),
                power=power,
                mdi_power=at_power,
                mdi_power_pp=maybe_pp(at_power, scale),
                mdi_power_smoothed=at_power,
                mdi_power_smoothed_pp=maybe_pp(at_power, scale),
                ratio_power=(at_power / value) if (at_power is not None and value) else None,
                pool_term=pool_term,
                pool_inflation=inflation,
                mdi_null_corrected=corrected,
                mdi_null_corrected_pp=to_pp(corrected, scale),
                abs_delta_quantile=abs_quantile,
                abs_delta_quantile_pp=to_pp(abs_quantile, scale),
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                ci_lo_pp=maybe_pp(ci_lo, scale),
                ci_hi_pp=maybe_pp(ci_hi, scale),
                mdi_null_in_sigma=(value / sigma) if sigma else None,
                sigma=sigma,
                n_draws=len(pooled),
                n_systems=len(clusters),
                n_items=max(item_counts),
                bootstrap=boot[index]["meta"],
            )
        )
    return entries


def supported_sweep(
    all_records: Iterable[ScoreRecord],
    env_id: str,
    *,
    split: RepeatSplit,
    sweep: Sequence[int] = DEFAULT_SWEEP,
) -> List[int]:
    """The sweep values *env_id*'s estimation repeats can support (flagged default #2)."""
    env_records = rec.filter_records(all_records, env_id=env_id)
    cells = rec.group_by_cell(env_records, repeats=list(split["estimation"]))
    if not cells:
        raise ValueError(
            f"env_id={env_id!r}: no scored cell within the estimation repeats {split['estimation']}"
        )
    supported_repeats = min(len(cell) for cell in cells.values())
    return sorted({n for n in sweep if 1 <= n <= supported_repeats})


def _apply_isotonic(entries: List[NullMdiEntry]) -> None:
    """Overwrite the smoothed fields per ``(env, alpha)`` with the PAVA sequence over N.

    The power column is smoothed on the same principle but only over the budgets
    where it is defined: an ``N`` whose grid did not reach *power* stays ``None``
    rather than borrowing a neighbour's value (ADR-018 — no extrapolation).
    """
    groups: Dict[Tuple[str, float], List[NullMdiEntry]] = {}
    for entry in entries:
        groups.setdefault((entry["env_id"], entry["alpha"]), []).append(entry)
    for key in sorted(groups):
        sequence = sorted(groups[key], key=lambda entry: entry["n_repeats"])
        smoothed = isotonic_non_increasing([entry["mdi_null"] for entry in sequence])
        for entry, value in zip(sequence, smoothed):
            entry["mdi_null_smoothed"] = value
            entry["mdi_null_smoothed_pp"] = to_pp(value, entry["scale"])
        defined = [entry for entry in sequence if entry["mdi_power"] is not None]
        powers = isotonic_non_increasing(
            [entry["mdi_power"] for entry in defined]  # type: ignore[misc]
        )
        for entry, power_value in zip(defined, powers):
            entry["mdi_power_smoothed"] = power_value
            entry["mdi_power_smoothed_pp"] = to_pp(power_value, entry["scale"])


def null_mdi_entries(
    all_records: Iterable[ScoreRecord],
    *,
    split: RepeatSplit,
    env_ids: Optional[Sequence[str]] = None,
    sweep: Sequence[int] = DEFAULT_SWEEP,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    seed: int = DEFAULT_SEED,
    draws_per_system: int = DEFAULT_DRAWS_PER_SYSTEM,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    power: float = DEFAULT_POWER,
) -> List[NullMdiEntry]:
    """Primary-path MDI table: every ``(env, supported N, alpha)``, isotonic-smoothed over N.

    The seed is offset deterministically per ``(env, N)`` (plus
    :data:`NULL_SEED_OFFSET` to keep the stream disjoint from the FIP curves),
    so the table is reproducible and each cell is independent.
    """
    ordered = rec.sorted_records(all_records)
    targets = sorted(env_ids) if env_ids is not None else rec.env_ids(ordered)
    entries: List[NullMdiEntry] = []
    for env_index, env_id in enumerate(targets):
        supported = supported_sweep(ordered, env_id, split=split, sweep=sweep)
        for n_index, n_repeats in enumerate(supported):
            entries.extend(
                estimate_null_mdi(
                    ordered,
                    env_id,
                    split=split,
                    n_repeats=n_repeats,
                    alphas=alphas,
                    seed=seed + NULL_SEED_OFFSET + 1009 * env_index + 17 * n_index,
                    draws_per_system=draws_per_system,
                    n_resamples=n_resamples,
                    ci_level=ci_level,
                    power=power,
                )
            )
    entries.sort(
        key=lambda entry: (
            entry["task"],
            entry["scale"],
            entry["env_id"],
            entry["n_repeats"],
            -entry["alpha"],
        )
    )
    _apply_isotonic(entries)
    return entries
