"""Exp 3 decay curves (FR-007): ``Var(mean_N) = sigma^2(env)/N + omega^2(env)``.

Miller's (2411.00640) MDE variance structure, re-read as a property of the
*measuring instrument* rather than of the sample (ADR-011 D3): repeating a judge
N times averages away the intrinsic sampling noise ``sigma^2``, but not the
procedural / systematic component ``omega^2`` — the **noise floor**. ADR-013
re-promotes this: omega^2 is not a finding, it is the parameter of the
self-improvement loop's confirmation condition.

Estimation, from bootstrap slices of the full-N cells (ADR-011 D1 — the
``N in {1,3,5,8,10,20}`` sweep is derived from the N=20 cells, never re-run):

1. One *evaluation round of budget N* = pick one procedural group (the
   ``group_field``, e.g. ``paraphrase_id``) uniformly at random from the cell,
   then draw N repeats with replacement from inside that group and average.
   Drawing inside a group is what leaves the group's offset uncancelled — that
   offset is exactly the floor the model calls omega^2.
2. ``Var_N(cell)`` = variance over those seeded draws; the curve point is the
   mean over cells (equal weight per cell).
3. Ordinary least squares of ``Var_N`` on ``1/N`` gives slope ``sigma^2`` and
   intercept ``omega^2``.

The N=1 point doubles as the reference for the theoretical ``1/sqrt(N)`` line;
``deviation`` is how far the measured spread sits above it (a pure-sampling
instrument would sit on it).

Methodology defaults flagged for first-author review (no Accepted ADR covers
them yet):

1. ``group_field`` defaults to ``paraphrase_id`` (ADR-011 D1's procedural axis).
   With a single group present, omega^2 estimates to ~0 by construction — that
   is the honest reading, not a bug.
2. ``omega_sq`` is the OLS intercept clamped at 0; ``omega_sq_raw`` keeps the
   unclamped value. ``omega_sq_debiased`` additionally removes the
   ``((G-1)/G) * sigma^2 / m`` inflation that finite repeats-per-group inject
   into the between-group spread — reported alongside, never silently
   substituted.
3. ``icc`` is reported as ``omega^2 / (omega^2 + sigma^2)`` for ADR-011 D10's
   ``N_effective`` correction; the correction itself is not applied here.
4. :data:`DEFAULT_DRAWS_PER_CELL` and the equal weight per cell.
5. The confidence interval on ``omega^2`` (needed by the ADR-015c noise-floor
   display rule: a floor whose CI touches zero is rendered as "below detection",
   never as "omega = 0.000") is a seeded **cluster bootstrap over cells**: the
   per-cell ``Var_N`` vectors are resampled and the OLS intercept refitted per
   replicate, with the intercept clamped at zero exactly as the point estimate
   is. A single cell is degenerate and yields an undefined CI
   (:data:`mdi.stats.bootstrap.MIN_CLUSTERS_FOR_CI`).

Covered by an Accepted ADR, so **not** a default in the sense above:

- ADR-023 (Accepted 2026-08-06) adds a second interval on the two ADR-020
  components and their sum, resampling the ``group_field`` **labels** instead
  of cells or items. Every one of those quantities is a variance over that
  axis, so this is the interval carrying the "what if we had drawn other
  paraphrases?" uncertainty; the cell/item intervals are conditional on the
  observed label set and are kept unchanged beside it. With G clusters the
  percentile interval is biased low by the ``(G-1)/G`` factor, so ADR-023 §c
  binds callers: report ``n_group_clusters`` with it, never cite it below
  eight clusters, and never argue from two such intervals failing to overlap.
"""

import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, TypedDict

from mdi.stats import records as rec
from mdi.stats.bootstrap import (
    DEFAULT_CI_LEVEL,
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    BootstrapMeta,
    BootstrapResult,
    bootstrap_scalar,
    bootstrap_vector,
    make_rng,
)
from mdi.store import ScoreRecord

DEFAULT_REPEAT_SWEEP: Tuple[int, ...] = (1, 3, 5, 8, 10, 20)
"""ADR-011 D1 sweep extended with N=8 (ADR-013, first-author decision #5: VRR-Stop M=8)."""

DEFAULT_GROUP_FIELD: str = "paraphrase_id"
DEFAULT_DRAWS_PER_CELL: int = 400


class DecayPoint(TypedDict):
    """One measured point of the decay curve."""

    n: int
    var_mean: float
    sd_mean: float
    sd_theoretical: float
    deviation: float
    relative_deviation: Optional[float]
    n_cells: int


class DecayFit(TypedDict):
    """Fitted decay model for one environment."""

    env_id: str
    task: str
    scale: str
    sweep: List[int]
    points: List[DecayPoint]
    sigma_sq: float
    sigma: float
    omega_sq: float
    omega: float
    omega_sq_raw: float
    omega_sq_debiased: float
    omega_sq_ci_lo: Optional[float]
    omega_sq_ci_hi: Optional[float]
    omega_ci_hi: Optional[float]
    icc: Optional[float]
    r_squared: Optional[float]
    omega_sq_main: Optional[float]
    omega_sq_int: Optional[float]
    omega_sq_between: Optional[float]
    omega_sq_main_ci_lo: Optional[float]
    omega_sq_main_ci_hi: Optional[float]
    omega_sq_int_ci_lo: Optional[float]
    omega_sq_int_ci_hi: Optional[float]
    omega_sq_main_group_ci_lo: Optional[float]
    omega_sq_main_group_ci_hi: Optional[float]
    omega_sq_int_group_ci_lo: Optional[float]
    omega_sq_int_group_ci_hi: Optional[float]
    omega_sq_between_group_ci_lo: Optional[float]
    omega_sq_between_group_ci_hi: Optional[float]
    n_group_clusters: int
    pair_gap_shift_sd: Optional[float]
    pair_gap_shift_sd_max: Optional[float]
    pair_gap_shift_ci_lo: Optional[float]
    pair_gap_shift_ci_hi: Optional[float]
    n_pairs: int
    pair_gap_sign_flips: Optional[int]
    group_field: str
    mean_groups_per_cell: float
    mean_repeats_per_group: float
    n_cells: int
    draws_per_cell: int
    bootstrap: BootstrapMeta


class DecayReport(TypedDict):
    """Full FR-007 payload."""

    fits: List[DecayFit]


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean with ``math.fsum`` accumulation."""
    return math.fsum(values) / len(values)


def _population_variance(values: Sequence[float]) -> float:
    """Population variance (1/n divisor) — the spread of the drawn evaluation rounds."""
    mean = _mean(values)
    return math.fsum((value - mean) ** 2 for value in values) / len(values)


def grouped_cells(
    all_records: Iterable[ScoreRecord],
    env_id: str,
    *,
    group_field: str = DEFAULT_GROUP_FIELD,
    min_repeats: int = 2,
) -> List[List[List[float]]]:
    """Cell -> group -> scores, all sorted; groups with < *min_repeats* scores are dropped."""
    env_records = rec.filter_records(all_records, env_id=env_id)
    buckets: Dict[Tuple[str, str, str], Dict[str, Dict[int, float]]] = {}
    for record in rec.scored_records(env_records):
        key = (record["env_id"], record["item_id"], record["system_id"])
        group = str(record[group_field])  # type: ignore[literal-required]
        score = record["parsed_score"]
        assert score is not None
        buckets.setdefault(key, {}).setdefault(group, {})[record["repeat_idx"]] = score

    cells: List[List[List[float]]] = []
    for key in sorted(buckets):
        groups = [
            [values[idx] for idx in sorted(values)]
            for _, values in sorted(buckets[key].items())
            if len(values) >= min_repeats
        ]
        if groups:
            cells.append(groups)
    return cells


def keyed_group_means(
    all_records: Iterable[ScoreRecord],
    env_id: str,
    *,
    group_field: str = DEFAULT_GROUP_FIELD,
    min_repeats: int = 2,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """``item -> system -> group -> mean score``, sorted, groups under *min_repeats* dropped.

    :func:`grouped_cells` throws the item/system keys away, which is fine for
    the decay fit but not for the ADR-020 decomposition: telling a common
    prompt shift apart from a system-specific one needs to know which system a
    cell belongs to.
    """
    env_records = rec.filter_records(all_records, env_id=env_id)
    buckets: Dict[str, Dict[str, Dict[str, Dict[int, float]]]] = {}
    for record in rec.scored_records(env_records):
        group = str(record[group_field])  # type: ignore[literal-required]
        score = record["parsed_score"]
        assert score is not None
        item = buckets.setdefault(record["item_id"], {})
        system = item.setdefault(record["system_id"], {})
        system.setdefault(group, {})[record["repeat_idx"]] = score

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for item_id in sorted(buckets):
        systems: Dict[str, Dict[str, float]] = {}
        for system_id in sorted(buckets[item_id]):
            groups = {
                group: _mean([values[idx] for idx in sorted(values)])
                for group, values in sorted(buckets[item_id][system_id].items())
                if len(values) >= min_repeats
            }
            if groups:
                systems[system_id] = groups
        if systems:
            out[item_id] = systems
    return out


def common_groups(
    items: Sequence[Tuple[str, Dict[str, Dict[str, float]]]],
) -> List[str]:
    """Group labels present in every usable cell — the cluster unit of the procedural axis.

    ADR-023: the ``omega^2`` components are variances *over this axis*, so an
    honest interval resamples these labels rather than the cells that happen to
    carry them. Cells short of two systems or two groups identify neither
    component and are excluded from the intersection, exactly as
    :func:`omega_components` excludes them.
    """
    common: Optional[Set[str]] = None
    for _item_id, systems in items:
        usable = {s: g for s, g in systems.items() if len(g) >= 2}
        if len(usable) < 2:
            continue
        groups = set.intersection(*(set(g) for g in usable.values()))
        if len(groups) < 2:
            continue
        common = groups if common is None else (common & groups)
    return sorted(common) if common else []


def pair_gap_shift(
    items: Sequence[Tuple[str, Dict[str, Dict[str, float]]]],
    groups_override: Optional[Sequence[str]] = None,
) -> Optional[Tuple[float, float, int, int]]:
    """How far a *system pair's item-averaged gap* travels when the group changes.

    ``omega_sq_int`` is a per-cell quantity, but a leaderboard gap is an average
    over items, where the item-specific part of the interaction washes out. This
    estimator measures the surviving part with no model in the way: average each
    system's score over items *within* one group, difference two systems, and
    look at the spread of that difference across groups.

    Returns ``(median_sd, max_sd, n_pairs, n_sign_flips)`` over the system pairs,
    in raw score units, or ``None`` when no pair is measurable. ``n_sign_flips``
    counts pairs whose gap changes sign across groups — an ordering that the
    prompt wording alone can reverse.
    """
    groups = list(groups_override) if groups_override is not None else common_groups(items)
    if len(groups) < 2:
        return None
    totals: Dict[str, Dict[str, List[float]]] = {}
    for _item_id, systems in items:
        usable = {s: g for s, g in systems.items() if set(groups) <= set(g)}
        if len(usable) < 2:
            continue
        for system_id, cell in usable.items():
            per_group = totals.setdefault(system_id, {})
            for group in groups:
                per_group.setdefault(group, []).append(cell[group])
    shared = sorted(s for s, g in totals.items() if all(g.get(k) for k in groups))
    if len(shared) < 2:
        return None
    means = {s: [_mean(totals[s][g]) for g in groups] for s in shared}
    sds: List[float] = []
    flips = 0
    for index, left in enumerate(shared):
        for right in shared[index + 1 :]:
            gaps = [a - b for a, b in zip(means[left], means[right])]
            sds.append(math.sqrt(_population_variance(gaps)))
            if min(gaps) < 0.0 < max(gaps):
                flips += 1
    ordered = sorted(sds)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else _mean([ordered[middle - 1], ordered[middle]])
    return median, max(ordered), len(ordered), flips


def omega_components(
    items: Sequence[Tuple[str, Dict[str, Dict[str, float]]]],
    groups_override: Optional[Sequence[str]] = None,
) -> Optional[Tuple[float, float]]:
    """Split the between-group floor into ``(main, interaction)`` (ADR-020 §a).

    For each item ``i``, system ``s`` and group ``p``, write the cell mean's
    deviation from its own cell average as ``d[i,s,p]``. The part shared by
    every system in that item, ``alpha[i,p] = mean_s d[i,s,p]``, is what
    cancels when two systems are compared under the *same* prompt; the residual
    ``beta[i,s,p]`` is what survives. Because ``beta`` sums to zero over
    systems the two variances add up exactly, so
    ``omega_sq_main + omega_sq_int`` reconstructs the between-group variance.

    Items carrying fewer than two systems or two groups are skipped: neither
    component is identified there. Returns ``None`` when nothing is usable.

    *groups_override* fixes the group list — repeats allowed — instead of taking
    each item's own intersection. The paraphrase-cluster bootstrap (ADR-023)
    passes a resampled label multiset through it; items missing any requested
    label are skipped, so a resample never silently falls back to a different
    axis.
    """
    main_terms: List[float] = []
    int_terms: List[float] = []
    for _item_id, systems in items:
        usable = {s: g for s, g in systems.items() if len(g) >= 2}
        if len(usable) < 2:
            continue
        if groups_override is None:
            groups = sorted(set.intersection(*(set(g) for g in usable.values())))
        else:
            available = set.intersection(*(set(g) for g in usable.values()))
            if not set(groups_override) <= available:
                continue
            groups = list(groups_override)
        if len(groups) < 2:
            continue
        deviations: Dict[str, List[float]] = {}
        for system_id in sorted(usable):
            means = [usable[system_id][g] for g in groups]
            centre = _mean(means)
            deviations[system_id] = [value - centre for value in means]
        alpha = [_mean([deviations[s][k] for s in sorted(deviations)]) for k in range(len(groups))]
        main_terms.append(_population_variance(alpha))
        for system_id in sorted(deviations):
            beta = [deviations[system_id][k] - alpha[k] for k in range(len(groups))]
            int_terms.append(_population_variance(beta))
    if not main_terms or not int_terms:
        return None
    return _mean(main_terms), _mean(int_terms)


def _draw_variance(
    groups: Sequence[Sequence[float]],
    n_repeats: int,
    draws: int,
    rng: random.Random,
) -> float:
    """Variance of *draws* evaluation rounds of budget *n_repeats* over one cell."""
    means: List[float] = []
    for _ in range(draws):
        group = groups[rng.randrange(len(groups))]
        means.append(
            math.fsum(group[rng.randrange(len(group))] for _ in range(n_repeats)) / n_repeats
        )
    return _population_variance(means)


def _ols(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, Optional[float]]:
    """Least-squares fit ``y = slope * x + intercept``; returns (slope, intercept, r^2)."""
    if len(xs) < 2:
        raise ValueError("the decay fit needs at least two distinct N values")
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    sxx = math.fsum((x - mean_x) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("the decay fit needs at least two distinct N values")
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    syy = math.fsum((y - mean_y) ** 2 for y in ys)
    residual = math.fsum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r_squared = None if syy == 0 else 1.0 - residual / syy
    return slope, intercept, r_squared


def fit_decay(
    all_records: Iterable[ScoreRecord],
    env_id: str,
    *,
    sweep: Sequence[int] = DEFAULT_REPEAT_SWEEP,
    group_field: str = DEFAULT_GROUP_FIELD,
    seed: int = DEFAULT_SEED,
    draws_per_cell: int = DEFAULT_DRAWS_PER_CELL,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> DecayFit:
    """Fit ``Var(mean_N) = sigma^2/N + omega^2`` for one environment.

    *sweep* is filtered to the N values the data supports (a cell must hold at
    least two repeats in some group); at least two distinct N values must
    survive or the fit raises. ``omega^2`` additionally carries a seeded
    cluster-bootstrap CI over cells (module docstring, flagged default #5).
    """
    env_records = rec.filter_records(all_records, env_id=env_id)
    if not env_records:
        raise ValueError(f"no records for env_id={env_id!r}")
    rec.check_schema(env_records)
    meta = {m["env_id"]: m for m in rec.env_metadata(env_records)}[env_id]

    cells = grouped_cells(env_records, env_id, group_field=group_field)
    if not cells:
        raise ValueError(f"env_id={env_id!r}: no cell has a {group_field} group with >= 2 repeats")
    max_repeats = max(len(group) for groups in cells for group in groups)
    supported = sorted({n for n in sweep if 1 <= n <= max_repeats})
    if len(supported) < 2:
        raise ValueError(
            f"env_id={env_id!r}: the decay sweep needs >= 2 distinct N values supported by "
            f"the data (largest {group_field} group holds {max_repeats} repeats; "
            f"requested sweep {sorted(set(sweep))})"
        )

    rng = make_rng(seed)
    per_cell: List[List[float]] = [[] for _ in cells]
    points_raw: List[Tuple[int, float]] = []
    for n_repeats in supported:
        for index, groups in enumerate(cells):
            per_cell[index].append(_draw_variance(groups, n_repeats, draws_per_cell, rng))
        points_raw.append((n_repeats, _mean([row[-1] for row in per_cell])))

    xs = [1.0 / n for n, _ in points_raw]
    ys = [variance for _, variance in points_raw]
    sigma_sq, intercept, r_squared = _ols(xs, ys)
    sigma_sq = max(sigma_sq, 0.0)
    omega_sq = max(intercept, 0.0)

    def _omega_sq_statistic(sample: Sequence[List[float]]) -> Optional[float]:
        """Clamped OLS intercept over a cell resample (same fit as the point estimate)."""
        means = [_mean([row[index] for row in sample]) for index in range(len(supported))]
        _, resampled_intercept, _ = _ols(xs, means)
        return max(resampled_intercept, 0.0)

    omega_boot = bootstrap_scalar(
        per_cell,
        _omega_sq_statistic,
        seed=seed,
        n_resamples=n_resamples,
        ci_level=ci_level,
    )

    group_counts = [len(groups) for groups in cells]
    repeat_counts = [len(group) for groups in cells for group in groups]
    mean_groups = _mean([float(count) for count in group_counts])
    mean_repeats = _mean([float(count) for count in repeat_counts])
    finite_group_inflation = 0.0
    if mean_groups > 1.0 and mean_repeats > 0.0:
        finite_group_inflation = ((mean_groups - 1.0) / mean_groups) * sigma_sq / mean_repeats
    omega_sq_debiased = max(omega_sq - finite_group_inflation, 0.0)

    sd_reference = math.sqrt(max(ys[0], 0.0)) * math.sqrt(supported[0])
    points: List[DecayPoint] = []
    for n_repeats, variance in points_raw:
        sd = math.sqrt(max(variance, 0.0))
        theoretical = sd_reference / math.sqrt(n_repeats)
        points.append(
            DecayPoint(
                n=n_repeats,
                var_mean=variance,
                sd_mean=sd,
                sd_theoretical=theoretical,
                deviation=sd - theoretical,
                relative_deviation=None if theoretical == 0 else (sd - theoretical) / theoretical,
                n_cells=len(cells),
            )
        )

    # ADR-020: split the between-group floor into the part that cancels in an
    # A-vs-B difference under a fixed prompt (main) and the part that does not
    # (interaction). Bootstrapped over the same cluster unit as omega itself --
    # here the item, since the split is defined within an item across systems.
    keyed = keyed_group_means(env_records, env_id, group_field=group_field)
    keyed_items = [(item_id, keyed[item_id]) for item_id in sorted(keyed)]
    components = omega_components(keyed_items)
    omega_sq_main: Optional[float] = None
    omega_sq_int: Optional[float] = None
    omega_sq_between: Optional[float] = None
    main_boot = int_boot = None
    if components is not None:
        omega_sq_main, omega_sq_int = components
        omega_sq_between = omega_sq_main + omega_sq_int
        main_boot = bootstrap_scalar(
            keyed_items,
            lambda sample: (lambda c: None if c is None else c[0])(omega_components(sample)),
            seed=seed + 1,
            n_resamples=n_resamples,
            ci_level=ci_level,
        )
        int_boot = bootstrap_scalar(
            keyed_items,
            lambda sample: (lambda c: None if c is None else c[1])(omega_components(sample)),
            seed=seed + 2,
            n_resamples=n_resamples,
            ci_level=ci_level,
        )

    # ADR-023: the intervals above resample cells (omega) and items (the two
    # components), but every one of these quantities is a variance *over the
    # procedural axis*. Resampling the axis itself is the interval that answers
    # "what if we had drawn a different set of paraphrases?" -- the question the
    # numbers are actually used to answer. Reported alongside the cell/item
    # intervals, never substituted for them.
    labels = common_groups(keyed_items)
    pp_boot: Optional[List[BootstrapResult]] = None
    if len(labels) >= 2:

        def _components_over(sample: Sequence[str]) -> Sequence[Optional[float]]:
            """(main, interaction, between) for one paraphrase-label resample."""
            parts = omega_components(keyed_items, sample)
            if parts is None:
                return (None, None, None)
            return (parts[0], parts[1], parts[0] + parts[1])

        pp_boot = bootstrap_vector(
            labels,
            _components_over,
            seed=seed + 3,
            n_resamples=n_resamples,
            ci_level=ci_level,
        )

    # The item-averaged counterpart of omega_sq_int: how far a leaderboard gap
    # travels when only the prompt is swapped. This is the quantity a table of
    # per-environment thresholds has to be read against, since those thresholds
    # hold the prompt fixed.
    shift = pair_gap_shift(keyed_items)
    shift_boot: Optional[BootstrapResult] = None
    if shift is not None and len(labels) >= 2:
        shift_boot = bootstrap_scalar(
            labels,
            lambda sample: (lambda v: None if v is None else v[0])(
                pair_gap_shift(keyed_items, sample)
            ),
            seed=seed + 4,
            n_resamples=n_resamples,
            ci_level=ci_level,
        )

    total = sigma_sq + omega_sq
    omega_sq_ci_hi = omega_boot["ci_hi"]
    return DecayFit(
        env_id=env_id,
        task=meta["task"],
        scale=meta["scale"],
        sweep=supported,
        points=points,
        sigma_sq=sigma_sq,
        sigma=math.sqrt(sigma_sq),
        omega_sq=omega_sq,
        omega=math.sqrt(omega_sq),
        omega_sq_raw=intercept,
        omega_sq_debiased=omega_sq_debiased,
        omega_sq_ci_lo=omega_boot["ci_lo"],
        omega_sq_ci_hi=omega_sq_ci_hi,
        omega_ci_hi=None if omega_sq_ci_hi is None else math.sqrt(max(omega_sq_ci_hi, 0.0)),
        icc=None if total == 0 else omega_sq / total,
        r_squared=r_squared,
        omega_sq_main=omega_sq_main,
        omega_sq_int=omega_sq_int,
        omega_sq_between=omega_sq_between,
        omega_sq_main_ci_lo=None if main_boot is None else main_boot["ci_lo"],
        omega_sq_main_ci_hi=None if main_boot is None else main_boot["ci_hi"],
        omega_sq_int_ci_lo=None if int_boot is None else int_boot["ci_lo"],
        omega_sq_int_ci_hi=None if int_boot is None else int_boot["ci_hi"],
        omega_sq_main_group_ci_lo=None if pp_boot is None else pp_boot[0]["ci_lo"],
        omega_sq_main_group_ci_hi=None if pp_boot is None else pp_boot[0]["ci_hi"],
        omega_sq_int_group_ci_lo=None if pp_boot is None else pp_boot[1]["ci_lo"],
        omega_sq_int_group_ci_hi=None if pp_boot is None else pp_boot[1]["ci_hi"],
        omega_sq_between_group_ci_lo=None if pp_boot is None else pp_boot[2]["ci_lo"],
        omega_sq_between_group_ci_hi=None if pp_boot is None else pp_boot[2]["ci_hi"],
        n_group_clusters=len(labels),
        pair_gap_shift_sd=None if shift is None else shift[0],
        pair_gap_shift_sd_max=None if shift is None else shift[1],
        pair_gap_shift_ci_lo=None if shift_boot is None else shift_boot["ci_lo"],
        pair_gap_shift_ci_hi=None if shift_boot is None else shift_boot["ci_hi"],
        n_pairs=0 if shift is None else shift[2],
        pair_gap_sign_flips=None if shift is None else shift[3],
        group_field=group_field,
        mean_groups_per_cell=mean_groups,
        mean_repeats_per_group=mean_repeats,
        n_cells=len(cells),
        draws_per_cell=draws_per_cell,
        bootstrap=omega_boot["meta"],
    )


def decay_report(
    all_records: Iterable[ScoreRecord],
    *,
    env_ids: Optional[Sequence[str]] = None,
    sweep: Sequence[int] = DEFAULT_REPEAT_SWEEP,
    group_field: str = DEFAULT_GROUP_FIELD,
    seed: int = DEFAULT_SEED,
    draws_per_cell: int = DEFAULT_DRAWS_PER_CELL,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> DecayReport:
    """Fit the decay model for every requested environment (sorted, seeded per env)."""
    ordered = rec.sorted_records(all_records)
    targets = sorted(env_ids) if env_ids is not None else rec.env_ids(ordered)
    fits: List[DecayFit] = []
    for index, env_id in enumerate(targets):
        fits.append(
            fit_decay(
                ordered,
                env_id,
                sweep=sweep,
                group_field=group_field,
                seed=seed + 1009 * index,
                draws_per_cell=draws_per_cell,
                n_resamples=n_resamples,
                ci_level=ci_level,
            )
        )
    return DecayReport(fits=fits)
