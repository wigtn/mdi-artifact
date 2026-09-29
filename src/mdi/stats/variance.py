"""Exp 1 variance analysis (FR-005): sigma_judge, flip rates, Krippendorff's alpha.

Pure functions over raw score records — no network, no wall-clock, everything
sorted (AGENTS.md §3.3). Outputs per environment:

- ``sigma`` — the judge's within-cell scoring noise, pooled over all
  ``(env, item, system)`` cells with at least two repeats;
- ``flip_rate`` — item-level flip rate: the probability that two independent
  repeats of the *same* cell disagree;
- ``rank_flip_rate`` — the probability that a single-repeat comparison of two
  systems disagrees with their mean-over-all-repeats ordering;
- ``krippendorff_alpha`` — reliability coefficient reported for comparability
  with prior judge-reliability work (ADR-011 D6), repeats as "coders".

Methodology defaults flagged for first-author review (no Accepted ADR covers
them yet):

1. ``flip_rate`` is a **pairwise disagreement** rate over distinct repeat pairs
   within a cell (not "deviation from a modal score").
2. Ties count as a flip in :func:`rank_flip_rate`, matching the ``P(A <= B)``
   convention the PRD uses for FIP (PRD §2.2) — a tie is not evidence of the
   claimed ordering.
3. Krippendorff's alpha defaults to the **interval** difference metric, since
   the project's scales (likert5 / likert10 / score100) are analysed as numeric
   (ADR-001 Option A); ``nominal`` and ``ordinal`` are available for
   comparability with papers that report those.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

from mdi.stats import bootstrap
from mdi.stats import records as rec
from mdi.store import ScoreRecord

DEFAULT_SIGMA_RESAMPLES: int = 2000

INTERVAL: str = "interval"
NOMINAL: str = "nominal"
ORDINAL: str = "ordinal"
DEFAULT_ALPHA_LEVEL: str = INTERVAL


class EnvVariance(TypedDict):
    """Per-environment variance summary (one row of the judge x task x scale heatmap)."""

    env_id: str
    task: str
    scale: str
    judge_model: str
    temperature: float
    sigma: Optional[float]
    sigma_ci_lo: Optional[float]
    sigma_ci_hi: Optional[float]
    variance: Optional[float]
    mean_score: Optional[float]
    flip_rate: Optional[float]
    krippendorff_alpha: Optional[float]
    alpha_level: str
    n_cells: int
    n_scores: int
    parse_failure_rate: float
    sigma_bootstrap: Optional[bootstrap.BootstrapMeta]


class SystemVariance(TypedDict):
    """Per-system repeat instability inside one environment (heteroscedasticity check).

    ``mdi`` builds its null distribution by splitting one system's repeats in
    two, then averages that over systems. That construction represents a real
    A-vs-B comparison only if the systems have comparable spread: with strong
    heteroscedasticity the null would be centred on the wrong variance. These
    rows let a reader check the assumption instead of taking it on faith.
    """

    env_id: str
    system_id: str
    sigma: Optional[float]
    variance: Optional[float]
    mean_score: Optional[float]
    n_cells: int
    n_scores: int


class ItemFlipRate(TypedDict):
    """Item-level flip rate for one ``(env, item, system)`` cell."""

    env_id: str
    item_id: str
    system_id: str
    n_repeats: int
    n_pairs: int
    flip_rate: float
    sd: Optional[float]


class RankFlip(TypedDict):
    """Rank-flip rate between two systems inside one environment."""

    env_id: str
    system_a: str
    system_b: str
    n_items: int
    n_comparisons: int
    mean_delta: float
    reference_sign: int
    rank_flip_rate: float


class VarianceReport(TypedDict):
    """Full FR-005 output payload."""

    envs: List[EnvVariance]
    systems: List[SystemVariance]
    item_flip_rates: List[ItemFlipRate]
    rank_flips: List[RankFlip]
    alpha_level: str


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean with ``math.fsum`` accumulation (order-stable)."""
    return math.fsum(values) / len(values)


def cell_variance(values: Sequence[float]) -> Optional[float]:
    """Unbiased within-cell variance, or ``None`` for fewer than two repeats."""
    if len(values) < 2:
        return None
    mean = _mean(values)
    return math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)


def pooled_variance(cells: Iterable[Sequence[float]]) -> Optional[float]:
    """Pooled within-cell variance (sigma^2_judge) over all cells with >= 2 repeats."""
    numerator = 0.0
    denominator = 0
    for values in cells:
        if len(values) < 2:
            continue
        mean = _mean(values)
        numerator += math.fsum((value - mean) ** 2 for value in values)
        denominator += len(values) - 1
    if denominator == 0:
        return None
    return numerator / denominator


def pooled_sigma(cells: Iterable[Sequence[float]]) -> Optional[float]:
    """Pooled within-cell standard deviation (sigma_judge)."""
    variance = pooled_variance(cells)
    return None if variance is None else math.sqrt(variance)


def env_sigma(
    all_records: Iterable[ScoreRecord],
    env_id: str,
    *,
    repeats: Optional[Sequence[int]] = None,
) -> Optional[float]:
    """sigma_judge for one environment, optionally restricted to *repeats* (ADR-007 split)."""
    env_records = rec.filter_records(all_records, env_id=env_id)
    cells = rec.group_by_cell(env_records, repeats=repeats)
    return pooled_sigma([rec.cell_values(cell) for cell in cells.values()])


def cell_flip_rate(values: Sequence[float]) -> Optional[float]:
    """Probability that two distinct repeats of one cell disagree (pairwise disagreement)."""
    n = len(values)
    if n < 2:
        return None
    disagreements = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if values[i] != values[j]:
                disagreements += 1
    return disagreements / total


def item_flip_rates(
    all_records: Iterable[ScoreRecord],
    *,
    repeats: Optional[Sequence[int]] = None,
) -> List[ItemFlipRate]:
    """Item-level flip rate per ``(env, item, system)`` cell, sorted by cell key."""
    cells = rec.group_by_cell(all_records, repeats=repeats)
    out: List[ItemFlipRate] = []
    for key in sorted(cells):
        values = rec.cell_values(cells[key])
        flip = cell_flip_rate(values)
        if flip is None:
            continue
        n = len(values)
        out.append(
            ItemFlipRate(
                env_id=key.env_id,
                item_id=key.item_id,
                system_id=key.system_id,
                n_repeats=n,
                n_pairs=n * (n - 1) // 2,
                flip_rate=flip,
                sd=math.sqrt(cell_variance(values) or 0.0),
            )
        )
    return out


def rank_flip_rate(
    all_records: Iterable[ScoreRecord],
    env_id: str,
    system_a: str,
    system_b: str,
    *,
    repeats: Optional[Sequence[int]] = None,
) -> Optional[RankFlip]:
    """Rank-flip rate between two systems in one environment.

    Reference ordering: the sign of the mean-over-all-repeats score difference,
    averaged over the items both systems were scored on. Each ``(repeat of A,
    repeat of B)`` combination on an item is one independent single-run
    comparison; the flip rate is the share of those comparisons whose sign
    contradicts the reference (ties count as flips, see the module docstring).
    Returns ``None`` when the two systems share no scored items.
    """
    env_records = rec.filter_records(all_records, env_id=env_id, system_ids=[system_a, system_b])
    cells = rec.group_by_cell(env_records, repeats=repeats)
    items_a = {key.item_id for key in cells if key.system_id == system_a}
    items_b = {key.item_id for key in cells if key.system_id == system_b}
    items = sorted(items_a & items_b)
    if not items:
        return None

    item_deltas: List[float] = []
    comparisons = 0
    flips = 0
    per_item_values: List[Tuple[List[float], List[float]]] = []
    for item_id in items:
        values_a = rec.cell_values(cells[rec.CellKey(env_id, item_id, system_a)])
        values_b = rec.cell_values(cells[rec.CellKey(env_id, item_id, system_b)])
        item_deltas.append(_mean(values_a) - _mean(values_b))
        per_item_values.append((values_a, values_b))

    mean_delta = _mean(item_deltas)
    reference = 0 if mean_delta == 0 else (1 if mean_delta > 0 else -1)
    for values_a, values_b in per_item_values:
        for value_a in values_a:
            for value_b in values_b:
                comparisons += 1
                difference = value_a - value_b
                sign = 0 if difference == 0 else (1 if difference > 0 else -1)
                if sign != reference or sign == 0:
                    flips += 1
    return RankFlip(
        env_id=env_id,
        system_a=system_a,
        system_b=system_b,
        n_items=len(items),
        n_comparisons=comparisons,
        mean_delta=mean_delta,
        reference_sign=reference,
        rank_flip_rate=flips / comparisons if comparisons else 0.0,
    )


def _value_counts(values: Iterable[float]) -> Dict[float, int]:
    """Multiset of values as ``{value: count}``."""
    counts: Dict[float, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _ordinal_deltas(counts: Dict[float, int]) -> Dict[Tuple[float, float], float]:
    """Krippendorff's ordinal difference metric for every value pair in *counts*."""
    ordered = sorted(counts)
    cumulative: Dict[float, float] = {}
    running = 0.0
    for value in ordered:
        running += counts[value]
        cumulative[value] = running
    deltas: Dict[Tuple[float, float], float] = {}
    for left in ordered:
        for right in ordered:
            low, high = (left, right) if left <= right else (right, left)
            span = cumulative[high] - cumulative[low] + counts[low]
            deltas[(left, right)] = (span - (counts[low] + counts[high]) / 2.0) ** 2
    return deltas


def _disagreement_sum(
    counts: Dict[float, int], level: str, global_counts: Dict[float, int]
) -> float:
    """Sum of delta^2 over all ordered value pairs of the multiset *counts*."""
    if level == INTERVAL:
        n = sum(counts.values())
        sum_v = math.fsum(value * count for value, count in counts.items())
        sum_v2 = math.fsum(value * value * count for value, count in counts.items())
        return 2.0 * n * sum_v2 - 2.0 * sum_v * sum_v
    if level == NOMINAL:
        n = sum(counts.values())
        same = math.fsum(count * count for count in counts.values())
        return float(n * n) - same
    if level == ORDINAL:
        deltas = _ordinal_deltas(global_counts)
        total = 0.0
        for left, count_left in counts.items():
            for right, count_right in counts.items():
                total += count_left * count_right * deltas[(left, right)]
        return total
    raise ValueError(f"unknown alpha level: {level!r} (expected interval/nominal/ordinal)")


def krippendorff_alpha(
    units: Sequence[Sequence[float]],
    level: str = DEFAULT_ALPHA_LEVEL,
) -> Optional[float]:
    """Krippendorff's alpha over *units* (one unit = one cell's repeated scores).

    ``alpha = 1 - D_o / D_e`` with the standard pairable-values formulation:
    units contribute their within-unit ordered value pairs to the observed
    disagreement, and the whole value pool contributes to the expected
    disagreement. Returns ``None`` when fewer than two pairable values exist or
    when ``D_e`` is zero (a degenerate, single-valued pool).
    """
    pairable = [list(unit) for unit in units if len(unit) >= 2]
    if not pairable:
        return None
    all_values = [value for unit in pairable for value in unit]
    n = len(all_values)
    if n < 2:
        return None
    global_counts = _value_counts(all_values)

    observed = 0.0
    for unit in pairable:
        counts = _value_counts(unit)
        observed += _disagreement_sum(counts, level, global_counts) / (len(unit) - 1)
    observed /= n
    expected = _disagreement_sum(global_counts, level, global_counts) / (n * (n - 1))
    if expected == 0:
        return None
    return 1.0 - observed / expected


def _pooled_sigma_over_cells(item_clusters: Sequence[List[List[float]]]) -> Optional[float]:
    """Pooled sigma over the cells of a list of item clusters (bootstrap statistic)."""
    return pooled_sigma([cell for cluster in item_clusters for cell in cluster])


def sigma_with_ci(
    env_cells_by_item: Dict[str, List[List[float]]],
    *,
    seed: int,
    n_resamples: int = DEFAULT_SIGMA_RESAMPLES,
    ci_level: float = bootstrap.DEFAULT_CI_LEVEL,
) -> bootstrap.BootstrapResult:
    """Pooled sigma_judge with a cluster bootstrap CI, resampling whole items.

    The dependence unit is the item: a cell is ``(item, system)``, so the
    scores within one item share that item's difficulty and are not
    exchangeable with another item's. Resampling whole items (not individual
    cells) keeps the interval honest. With fewer than two items the interval is
    reported as undefined rather than as a zero-width band (the shared
    :data:`mdi.stats.bootstrap.MIN_CLUSTERS_FOR_CI` guard).
    """
    clusters = [env_cells_by_item[item] for item in sorted(env_cells_by_item)]
    return bootstrap.bootstrap_scalar(
        clusters,
        _pooled_sigma_over_cells,
        seed=seed,
        n_resamples=n_resamples,
        ci_level=ci_level,
    )


def system_variance(
    all_records: Iterable[ScoreRecord],
    *,
    repeats: Optional[Sequence[int]] = None,
) -> List[SystemVariance]:
    """Per-``(env, system)`` pooled sigma, sorted — the input to the spread check."""
    ordered = rec.sorted_records(all_records)
    cells = rec.group_by_cell(ordered, repeats=repeats)
    grouped: Dict[Tuple[str, str], List[List[float]]] = {}
    for key in sorted(cells):
        grouped.setdefault((key.env_id, key.system_id), []).append(rec.cell_values(cells[key]))

    out: List[SystemVariance] = []
    for env_id, system_id in sorted(grouped):
        system_cells = grouped[(env_id, system_id)]
        flat = [value for values in system_cells for value in values]
        variance = pooled_variance(system_cells)
        out.append(
            SystemVariance(
                env_id=env_id,
                system_id=system_id,
                sigma=None if variance is None else math.sqrt(variance),
                variance=variance,
                mean_score=_mean(flat) if flat else None,
                n_cells=len(system_cells),
                n_scores=len(flat),
            )
        )
    return out


def env_variance(
    all_records: Iterable[ScoreRecord],
    *,
    alpha_level: str = DEFAULT_ALPHA_LEVEL,
    repeats: Optional[Sequence[int]] = None,
    seed: int = bootstrap.DEFAULT_SEED,
    sigma_resamples: int = DEFAULT_SIGMA_RESAMPLES,
) -> List[EnvVariance]:
    """Per-environment variance summary rows, sorted by ``env_id`` (FR-005 heatmap input)."""
    ordered = rec.sorted_records(all_records)
    rec.check_schema(ordered)
    metadata = {meta["env_id"]: meta for meta in rec.env_metadata(ordered)}
    cells = rec.group_by_cell(ordered, repeats=repeats)

    by_env: Dict[str, List[List[float]]] = {}
    by_env_item: Dict[str, Dict[str, List[List[float]]]] = {}
    for key in sorted(cells):
        values = rec.cell_values(cells[key])
        by_env.setdefault(key.env_id, []).append(values)
        by_env_item.setdefault(key.env_id, {}).setdefault(key.item_id, []).append(values)

    out: List[EnvVariance] = []
    for env_id in sorted(metadata):
        meta = metadata[env_id]
        env_cells = by_env.get(env_id, [])
        flat = [value for values in env_cells for value in values]
        variance = pooled_variance(env_cells)
        flips = [
            rate for rate in (cell_flip_rate(values) for values in env_cells) if rate is not None
        ]
        sigma_ci = sigma_with_ci(
            by_env_item.get(env_id, {}), seed=seed, n_resamples=sigma_resamples
        )
        out.append(
            EnvVariance(
                env_id=env_id,
                task=meta["task"],
                scale=meta["scale"],
                judge_model=meta["judge_model"],
                temperature=meta["temperature"],
                sigma=None if variance is None else math.sqrt(variance),
                sigma_ci_lo=sigma_ci["ci_lo"],
                sigma_ci_hi=sigma_ci["ci_hi"],
                variance=variance,
                mean_score=_mean(flat) if flat else None,
                flip_rate=_mean(flips) if flips else None,
                krippendorff_alpha=krippendorff_alpha(env_cells, alpha_level),
                alpha_level=alpha_level,
                n_cells=len(env_cells),
                n_scores=len(flat),
                parse_failure_rate=meta["parse_failure_rate"],
                sigma_bootstrap=sigma_ci["meta"],
            )
        )
    return out


def variance_report(
    all_records: Iterable[ScoreRecord],
    *,
    alpha_level: str = DEFAULT_ALPHA_LEVEL,
    repeats: Optional[Sequence[int]] = None,
    seed: int = bootstrap.DEFAULT_SEED,
    sigma_resamples: int = DEFAULT_SIGMA_RESAMPLES,
) -> VarianceReport:
    """Assemble the full FR-005 payload: env summaries, item flip rates, rank flips."""
    ordered = rec.sorted_records(all_records)
    envs = env_variance(
        ordered,
        alpha_level=alpha_level,
        repeats=repeats,
        seed=seed,
        sigma_resamples=sigma_resamples,
    )
    system_rows = system_variance(ordered, repeats=repeats)
    flip_rows = item_flip_rates(ordered, repeats=repeats)
    rank_rows: List[RankFlip] = []
    for env in envs:
        env_records = rec.filter_records(ordered, env_id=env["env_id"])
        systems = rec.system_ids(env_records)
        for index, system_a in enumerate(systems):
            for system_b in systems[index + 1 :]:
                flip = rank_flip_rate(
                    env_records, env["env_id"], system_a, system_b, repeats=repeats
                )
                if flip is not None:
                    rank_rows.append(flip)
    return VarianceReport(
        envs=envs,
        systems=system_rows,
        item_flip_rates=flip_rows,
        rank_flips=rank_rows,
        alpha_level=alpha_level,
    )
