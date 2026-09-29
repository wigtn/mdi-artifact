"""Exp 2 FIP estimation (FR-006): P(the observed gain reverses on re-evaluation).

``FIP(delta, env, N)`` is the probability that a gain of size ``delta``, observed
with an evaluation budget of ``N`` repeats in environment ``env``, reverses when
the *same* systems are evaluated again independently. Unlike a Type-S error rate
it conditions on the **observed** delta — the information a practitioner
actually holds (ADR-011 D5).

Estimation is nonparametric, entirely from measured repeats — no distributional
assumption, no extra API call:

1. Take the estimation half of the ADR-007 repeat split (the repeats used for
   *screening* close pairs are held out; estimating FIP on the repeats that
   selected the pairs is selection-on-noise and inflates it).
2. For each pair and each draw, split every cell's estimation repeats into two
   **disjoint** pools (a fresh random half/half split per draw), draw ``N``
   repeats with replacement from each, and average per item:
   ``delta_obs`` from the first pool, ``delta_re`` from the second. Disjoint
   pools make the re-evaluation independent of the observation by construction.
3. Orient each draw so the observed gain is positive, bin it by ``|delta_obs|``,
   and count reversals (``delta_re`` of the opposite sign, or exactly zero —
   the ``P(A <= B)`` convention of PRD §2.2).
4. Confidence intervals come from a cluster bootstrap over system pairs
   (:mod:`mdi.stats.bootstrap`).

Methodology defaults flagged for first-author review (no Accepted ADR covers
them yet):

1. **Bin rule** — default edges are ADR-007 §2's screening bins reused for the
   curve's x-axis: ``0, 0.5, 1, 2, 4`` in units of the environment's sigma, plus
   an open-ended top bin. ADR-007 §4 requires these to become config fields
   (``exp2.delta_bins``); until then they are the documented default.
2. **Ties count as a reversal** (``delta_re == 0``), per ``P(A <= B)``.
3. **Draws with ``delta_obs == 0`` are dropped** — there is no observed gain to
   reverse — and counted in the curve metadata.
4. **With-replacement draws inside each pool**, so a budget ``N`` larger than
   the available pool can still be simulated.
5. :data:`DEFAULT_DRAWS_PER_PAIR`, the fresh per-draw pool split, and pairs (not
   items) as the bootstrap cluster.
"""

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
    bootstrap_meta,
    bootstrap_vector,
    make_rng,
)
from mdi.stats.scales import maybe_pp, to_pp
from mdi.store import ScoreRecord

DEFAULT_SIGMA_BIN_MULTIPLES: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
DEFAULT_DRAWS_PER_PAIR: int = 200
DEFAULT_N_REPEATS: int = 1


class OverlappingRepeatsError(ValueError):
    """Screening and estimation repeat indices overlap — the ADR-007 guard."""


class RepeatSplit(TypedDict):
    """ADR-007 held-out split of ``repeat_idx`` values."""

    screening: List[int]
    estimation: List[int]


class FipBin(TypedDict):
    """One observed-delta bin of the FIP curve."""

    lo: float
    hi: Optional[float]
    mean_delta: Optional[float]
    n_draws: int
    n_reversals: int
    fip: Optional[float]
    ci_lo: Optional[float]
    ci_hi: Optional[float]


class FipMeta(TypedDict):
    """Everything needed to re-derive a curve, embedded next to it."""

    n_repeats: int
    sigma: float
    bin_edges: List[float]
    sigma_bin_multiples: List[float]
    draws_per_pair: int
    n_pairs: int
    n_items: int
    n_draws: int
    n_zero_delta_draws: int
    screening_repeats: List[int]
    estimation_repeats: List[int]
    tie_policy: str
    bootstrap: BootstrapMeta


class FipCurve(TypedDict):
    """FIP(delta | env, N) as a binned curve with per-bin bootstrap intervals."""

    env_id: str
    task: str
    scale: str
    n_repeats: int
    bins: List[FipBin]
    meta: FipMeta


class ConversionRow(TypedDict):
    """One row of the improvement-to-confidence conversion table (FR-006).

    ``*_pp`` fields are the ADR-001 primary notation (scale-normalized
    percentage points); the raw-scale values stay alongside them.
    """

    env_id: str
    task: str
    scale: str
    n_repeats: int
    delta_lo: float
    delta_hi: Optional[float]
    delta: Optional[float]
    delta_lo_pp: float
    delta_hi_pp: Optional[float]
    delta_pp: Optional[float]
    reversal_pct: Optional[float]
    ci_lo_pct: Optional[float]
    ci_hi_pct: Optional[float]
    n_draws: int
    statement: str


class FipReport(TypedDict):
    """Full FR-006 payload: one curve per (env, N) plus the conversion table."""

    curves: List[FipCurve]
    conversion_table: List[ConversionRow]


def make_split(screening: Sequence[int], estimation: Sequence[int]) -> RepeatSplit:
    """Build a validated ADR-007 repeat split (sorted, de-duplicated)."""
    split = RepeatSplit(screening=sorted(set(screening)), estimation=sorted(set(estimation)))
    validate_split(split)
    return split


def validate_split(split: RepeatSplit) -> None:
    """Raise :class:`OverlappingRepeatsError` if the split leaks screening into estimation.

    ADR-007: the repeats used to *select* close system pairs must be completely
    excluded from FIP estimation, otherwise regression to the mean inflates the
    curve ("selected on noise, evaluated with the same noise").
    """
    overlap = sorted(set(split["screening"]) & set(split["estimation"]))
    if overlap:
        raise OverlappingRepeatsError(
            "ADR-007 violation: screening and estimation repeat indices overlap at "
            f"{overlap} — FIP must not be estimated on the repeats used to select pairs"
        )
    if len(split["estimation"]) < 2:
        raise ValueError(
            "FIP needs at least two estimation repeats per cell to form disjoint "
            "observation / re-evaluation pools"
        )


def bin_edges_from_sigma(
    sigma: float,
    multiples: Sequence[float] = DEFAULT_SIGMA_BIN_MULTIPLES,
) -> List[float]:
    """Observed-delta bin edges in score units, from sigma multiples (ADR-007 §2)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive to build sigma-scaled bins, got {sigma}")
    edges = sorted({multiple * sigma for multiple in multiples})
    if edges[0] != 0.0:
        edges.insert(0, 0.0)
    return edges


def _bin_index(edges: Sequence[float], value: float) -> int:
    """Index of the half-open bin ``[edges[i], edges[i+1])`` containing *value* (last is open)."""
    for index in range(len(edges) - 1, -1, -1):
        if value >= edges[index]:
            return index
    return 0


def draw_mean(pool: Sequence[float], n_repeats: int, rng: random.Random) -> float:
    """Mean of *n_repeats* draws with replacement from *pool*.

    Shared with :mod:`mdi.stats.null_mdi` so the primary (null-PI) and
    validation (FIP-crossing) MDI paths use identical draw machinery (ADR-015b).
    """
    return math.fsum(pool[rng.randrange(len(pool))] for _ in range(n_repeats)) / n_repeats


def split_pools(
    values: Dict[int, float],
    repeats: Sequence[int],
    rng: random.Random,
) -> Tuple[List[float], List[float]]:
    """Randomly split a cell's estimation repeats into two disjoint score pools.

    Shared with :mod:`mdi.stats.null_mdi` (see :func:`draw_mean`).
    """
    available = [idx for idx in repeats if idx in values]
    shuffled = list(available)
    rng.shuffle(shuffled)
    half = len(shuffled) // 2
    observed = [values[idx] for idx in shuffled[:half]]
    re_evaluated = [values[idx] for idx in shuffled[half:]]
    return observed, re_evaluated


class _PairDraws(TypedDict):
    """Bootstrap cluster: every draw belonging to one system pair."""

    pair: Tuple[str, str]
    draws: List[Tuple[float, bool]]


def _pair_draws(
    cells: Dict[rec.CellKey, Dict[int, float]],
    env_id: str,
    system_a: str,
    system_b: str,
    items: Sequence[str],
    estimation: Sequence[int],
    n_repeats: int,
    draws_per_pair: int,
    rng: random.Random,
) -> List[Tuple[float, bool]]:
    """Draw ``(|delta_obs|, reversed)`` observations for one system pair."""
    out: List[Tuple[float, bool]] = []
    for _ in range(draws_per_pair):
        observed_delta = 0.0
        re_delta = 0.0
        for item_id in items:
            values_a = cells[rec.CellKey(env_id, item_id, system_a)]
            values_b = cells[rec.CellKey(env_id, item_id, system_b)]
            obs_pool_a, re_pool_a = split_pools(values_a, estimation, rng)
            obs_pool_b, re_pool_b = split_pools(values_b, estimation, rng)
            observed_delta += draw_mean(obs_pool_a, n_repeats, rng) - draw_mean(
                obs_pool_b, n_repeats, rng
            )
            re_delta += draw_mean(re_pool_a, n_repeats, rng) - draw_mean(re_pool_b, n_repeats, rng)
        observed_delta /= len(items)
        re_delta /= len(items)
        if observed_delta == 0.0:
            continue
        oriented = re_delta if observed_delta > 0 else -re_delta
        out.append((abs(observed_delta), oriented <= 0.0))
    return out


def _bin_rates(
    clusters: Sequence[_PairDraws],
    edges: Sequence[float],
) -> List[Optional[float]]:
    """Per-bin reversal rate over the draws of *clusters* (``None`` for an empty bin)."""
    totals = [0] * len(edges)
    reversals = [0] * len(edges)
    for cluster in clusters:
        for delta, reversed_flag in cluster["draws"]:
            index = _bin_index(edges, delta)
            totals[index] += 1
            if reversed_flag:
                reversals[index] += 1
    return [(reversals[i] / totals[i]) if totals[i] else None for i in range(len(edges))]


def system_pairs(all_records: Iterable[ScoreRecord], env_id: str) -> List[Tuple[str, str]]:
    """All unordered system pairs present in *env_id*, sorted (default when none are given)."""
    systems = rec.system_ids(rec.filter_records(all_records, env_id=env_id))
    return [(a, b) for index, a in enumerate(systems) for b in systems[index + 1 :]]


def estimate_fip(
    all_records: Iterable[ScoreRecord],
    env_id: str,
    *,
    split: RepeatSplit,
    n_repeats: int = DEFAULT_N_REPEATS,
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    seed: int = DEFAULT_SEED,
    bin_edges: Optional[Sequence[float]] = None,
    sigma_bin_multiples: Sequence[float] = DEFAULT_SIGMA_BIN_MULTIPLES,
    draws_per_pair: int = DEFAULT_DRAWS_PER_PAIR,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> FipCurve:
    """Estimate ``FIP(delta | env_id, n_repeats)`` from measured repeats.

    *split* must honour ADR-007 (screening repeats disjoint from estimation
    repeats); overlapping indices raise :class:`OverlappingRepeatsError`.
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
    if sigma is None or sigma <= 0.0:
        raise ValueError(
            f"env_id={env_id!r}: sigma is not estimable from the estimation repeats "
            "(need >= 2 repeats in at least one cell, with some score variation)"
        )
    edges = (
        list(bin_edges)
        if bin_edges is not None
        else bin_edges_from_sigma(sigma, sigma_bin_multiples)
    )

    candidate_pairs = list(pairs) if pairs is not None else system_pairs(env_records, env_id)
    rng = make_rng(seed)
    clusters: List[_PairDraws] = []
    item_counts: List[int] = []
    zero_delta = 0
    for system_a, system_b in sorted(tuple(sorted(pair)) for pair in candidate_pairs):
        items = sorted(
            {key.item_id for key in cells if key.system_id == system_a}
            & {key.item_id for key in cells if key.system_id == system_b}
        )
        items = [
            item_id
            for item_id in items
            if len(cells[rec.CellKey(env_id, item_id, system_a)]) >= 2
            and len(cells[rec.CellKey(env_id, item_id, system_b)]) >= 2
        ]
        if not items:
            continue
        draws = _pair_draws(
            cells,
            env_id,
            system_a,
            system_b,
            items,
            estimation,
            n_repeats,
            draws_per_pair,
            rng,
        )
        zero_delta += draws_per_pair - len(draws)
        if not draws:
            continue
        item_counts.append(len(items))
        clusters.append(_PairDraws(pair=(system_a, system_b), draws=draws))

    if not clusters:
        raise ValueError(
            f"env_id={env_id!r}: no system pair has two systems scored on a shared item "
            "with >= 2 estimation repeats"
        )

    boot = bootstrap_vector(
        clusters,
        lambda sample: _bin_rates(sample, edges),
        seed=seed,
        n_resamples=n_resamples,
        ci_level=ci_level,
    )

    totals = [0] * len(edges)
    reversals = [0] * len(edges)
    delta_sums = [0.0] * len(edges)
    for cluster in clusters:
        for delta, reversed_flag in cluster["draws"]:
            index = _bin_index(edges, delta)
            totals[index] += 1
            delta_sums[index] += delta
            if reversed_flag:
                reversals[index] += 1

    bins: List[FipBin] = []
    for index, lo in enumerate(edges):
        hi: Optional[float] = edges[index + 1] if index + 1 < len(edges) else None
        count = totals[index]
        bins.append(
            FipBin(
                lo=lo,
                hi=hi,
                mean_delta=(delta_sums[index] / count) if count else None,
                n_draws=count,
                n_reversals=reversals[index],
                fip=(reversals[index] / count) if count else None,
                ci_lo=boot[index]["ci_lo"],
                ci_hi=boot[index]["ci_hi"],
            )
        )

    return FipCurve(
        env_id=env_id,
        task=meta["task"],
        scale=meta["scale"],
        n_repeats=n_repeats,
        bins=bins,
        meta=FipMeta(
            n_repeats=n_repeats,
            sigma=sigma,
            bin_edges=edges,
            sigma_bin_multiples=list(sigma_bin_multiples),
            draws_per_pair=draws_per_pair,
            n_pairs=len(clusters),
            n_items=max(item_counts) if item_counts else 0,
            n_draws=sum(totals),
            n_zero_delta_draws=zero_delta,
            screening_repeats=list(split["screening"]),
            estimation_repeats=estimation,
            tie_policy="tie_counts_as_reversal",
            bootstrap=bootstrap_meta(seed=seed, n_resamples=n_resamples, ci_level=ci_level),
        ),
    )


def conversion_table(curve: FipCurve) -> List[ConversionRow]:
    """ "A single-run delta = x reverses y% of the time" rows for one curve (FR-006).

    Statements quote the delta in ADR-001 %p notation first, with the raw-scale
    value kept in parentheses (and, untouched, in the row's ``delta`` fields).
    """
    label = "single-run" if curve["n_repeats"] == 1 else f"{curve['n_repeats']}-run"
    rows: List[ConversionRow] = []
    for entry in curve["bins"]:
        if entry["n_draws"] == 0 or entry["fip"] is None or entry["mean_delta"] is None:
            continue
        pct = 100.0 * entry["fip"]
        delta_pp = to_pp(entry["mean_delta"], curve["scale"])
        rows.append(
            ConversionRow(
                env_id=curve["env_id"],
                task=curve["task"],
                scale=curve["scale"],
                n_repeats=curve["n_repeats"],
                delta_lo=entry["lo"],
                delta_hi=entry["hi"],
                delta=entry["mean_delta"],
                delta_lo_pp=to_pp(entry["lo"], curve["scale"]),
                delta_hi_pp=maybe_pp(entry["hi"], curve["scale"]),
                delta_pp=delta_pp,
                reversal_pct=pct,
                ci_lo_pct=None if entry["ci_lo"] is None else 100.0 * entry["ci_lo"],
                ci_hi_pct=None if entry["ci_hi"] is None else 100.0 * entry["ci_hi"],
                n_draws=entry["n_draws"],
                statement=(
                    f"a {label} delta = {delta_pp:.1f} %p ({entry['mean_delta']:.3f} raw) on "
                    f"{curve['task']}/{curve['scale']} reverses {pct:.1f}% of the time"
                ),
            )
        )
    return rows


def fip_report(
    all_records: Iterable[ScoreRecord],
    *,
    split: RepeatSplit,
    env_ids: Optional[Sequence[str]] = None,
    n_repeats_sweep: Sequence[int] = (DEFAULT_N_REPEATS,),
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    seed: int = DEFAULT_SEED,
    draws_per_pair: int = DEFAULT_DRAWS_PER_PAIR,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    sigma_bin_multiples: Sequence[float] = DEFAULT_SIGMA_BIN_MULTIPLES,
) -> FipReport:
    """Estimate one curve per (env, N) and assemble the conversion table.

    The seed is offset deterministically per (env, N) so curves are independent
    yet reproducible.
    """
    ordered = rec.sorted_records(all_records)
    targets = list(env_ids) if env_ids is not None else rec.env_ids(ordered)
    curves: List[FipCurve] = []
    rows: List[ConversionRow] = []
    for env_index, env_id in enumerate(sorted(targets)):
        for n_index, n_repeats in enumerate(sorted(set(n_repeats_sweep))):
            curve = estimate_fip(
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
            curves.append(curve)
            rows.extend(conversion_table(curve))
    return FipReport(curves=curves, conversion_table=rows)
