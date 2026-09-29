"""External-oracle validation of the MDI threshold (ADR-029).

Everything else in this package measures the judge against itself: the null
distribution is two scorings of the *same* system, so it can only speak to the
instability channel. That leaves the question a reader actually cares about
untouched — whether a gain that clears the threshold corresponds to a real
quality difference — and leaves the ``(or real)`` in "gains that survive are bias
(or real)" unresolved.

The SummEval release pinned under ``data/inputs/`` answers it. Beside the machine
summaries this project scores, it carries three-expert annotations on four facets
for every summarizer. Until now those annotations were read only to *select*
close systems; here they are read to *check* the instrument.

Three products, all at zero API cost:

1. :func:`oracle_gaps` — the external gap between every scored pair, with a
   seeded item-cluster interval. Pairs whose interval covers zero are a null the
   estimator did not construct, so coverage measured on them is not circular
   (ADR-029b).
2. :func:`bias_decomposition` — ``judge gap = oracle gap + residual``. The
   residual is what repeats cannot remove and what a threshold cannot license
   (ADR-029c).
3. :func:`claim_rates` — how often ``|observed delta| > MDI`` fires on those
   externally-null pairs, per budget. Two readings are emitted: the *internal*
   rate (two scorings of one system — what MDI is built to control) and the
   *external* rate (two systems the oracle cannot separate). The gap between
   them is the measurement this module exists for, and the external rate is
   expected to *rise* with the budget wherever the judge orders systems
   systematically: more repeats buy more confidence in the same ordering,
   whether or not that ordering is right.

Scope, stated once. The oracle is itself a measurement — three raters, four
facets averaged with equal weight — so a pair its interval cannot separate is
"not separated by this reference at this precision", never "identical". Nothing
here licenses the sentence "the judge is wrong"; what it licenses is a
comparison of two instruments' resolutions on the same items.
"""

import itertools
import math
import random
import statistics
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

from mdi.stats import records as rec
from mdi.stats.bootstrap import (
    DEFAULT_CI_LEVEL,
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    BootstrapMeta,
    bootstrap_vector,
    make_rng,
)
from mdi.stats.fip import RepeatSplit, draw_mean, validate_split
from mdi.stats.scales import to_pp
from mdi.store import (
    SUMMEVAL_FACETS,
    SUMMEVAL_HUMAN_WIDTH,
    SUMMEVAL_SOURCE_SPEC,
    ScoreRecord,
    read_source_cache,
    summeval_human_scores,
)

#: Draws per pair when measuring how often a threshold fires. Matches the FIP
#: path's default so the two rest on the same sampling assumptions.
DEFAULT_DRAWS_PER_PAIR: int = 200
#: Deterministic offset keeping this module's RNG stream disjoint from the null
#: and FIP paths (which use ``seed + 1009*env + 17*n`` and an offset of 104729).
ORACLE_SEED_OFFSET: int = 15485863
#: External-gap boundary for the detection summary, in %p of the annotation
#: scale. Fixed by the first author (work order 2026-08-17, §4.1 item c): the
#: summary asks whether detection moves across a reference gap of this size the
#: way it moves across the threshold's own boundary. Descriptive reporting, not
#: a decision rule — MDI remains the only threshold anything is decided against.
EXTERNAL_BOUNDARY_PP: float = 2.0


class OracleGap(TypedDict):
    """The external gap between two scored systems, in %p of each instrument's scale."""

    system_a: str
    system_b: str
    n_items: int
    oracle_gap_pp: float
    oracle_ci_lo_pp: Optional[float]
    oracle_ci_hi_pp: Optional[float]
    oracle_t: Optional[float]
    #: ``True`` when the oracle interval excludes zero — the reference *can* tell
    #: these two apart. ``False`` pairs are the externally-null set.
    separated: bool
    judge_gap_pp: float
    #: ``judge_gap_pp - oracle_gap_pp``: the part of the observed ordering the
    #: external reference does not account for.
    residual_pp: float
    #: Facets whose single-facet interval excludes zero for this pair, sorted.
    #: Filled by :func:`facet_breakdown`; empty when no breakdown ran. A pair
    #: ``separated`` on the average but by NO single facet would be a pure
    #: aggregation artifact — the count of such pairs is the construct check.
    facets_separating: List[str]
    #: Signed per-facet gap in %p of the annotation scale. Same convention as
    #: ``oracle_gap_pp``; empty when no breakdown ran.
    facet_gaps_pp: Dict[str, float]
    bootstrap: BootstrapMeta


class ClaimRate(TypedDict):
    """How often a threshold fires, at one budget, on internally- vs externally-null pairs."""

    n_repeats: int
    alpha: float
    mdi_pp: float
    n_null_pairs: int
    n_draws: int
    #: Two scorings of the SAME system. This is the channel MDI is built to
    #: control, so it should sit near alpha; it is reported to show that it does.
    internal_rate: float
    #: Two DIFFERENT systems the oracle cannot separate. Nothing constrains this
    #: to alpha, and where the judge has a stable ordering it approaches one.
    external_rate: float


class DetectionPoint(TypedDict):
    """One pair's detection rate at one budget, against the gap the reference measures.

    :class:`ClaimRate` pools over the pairs the reference *cannot* separate, which
    answers "how often does the threshold fire when nothing is there". This is the
    other half (ADR-029 §e): one row per pair per budget, carrying the external gap
    alongside, so ``detect_rate`` can be read as a curve over an effect size the
    judge did not produce. On a zero-gap pair the row is a false-positive rate; on
    a wide pair it is power.
    """

    system_a: str
    system_b: str
    #: The reference's gap for this pair. The curve's x-axis.
    oracle_gap_pp: float
    #: Whether the reference's own interval excludes zero (see :class:`OracleGap`).
    separated: bool
    n_repeats: int
    mdi_pp: float
    n_draws: int
    #: P(|judge Δ| > MDI) over repeated scoring draws at this budget.
    detect_rate: float


class FacetReport(TypedDict):
    """The reference read as ONE facet: what a single annotation axis resolves.

    The main report averages the four facets with equal weight, matching the
    judge prompt. This row answers whether that average manufactures anything a
    single axis would not support, and how differently the axes resolve — the
    reference is itself not one instrument, and saying so is part of the result
    (ADR-033 construct check).
    """

    facet: str
    n_separated_pairs: int
    rank_concordance: Optional[float]
    oracle_spread_pp: float


class BoundarySummary(TypedDict):
    """Mean detection rate on either side of one boundary, at one budget.

    Two boundaries are summarized per budget. ``judge_gap_vs_mdi`` splits pairs
    by the judge's own gap against the threshold — detection is *defined* by
    that comparison, so a clean split here only confirms the design and is
    never a finding. ``oracle_gap_pp`` splits by the reference's gap against
    :data:`EXTERNAL_BOUNDARY_PP`; how little the rate moves across it is the
    measured content of ADR-029 §e.
    """

    n_repeats: int
    boundary: str
    boundary_value_pp: float
    n_below: int
    n_above: int
    mean_detect_below: Optional[float]
    mean_detect_above: Optional[float]


class SystemRow(TypedDict):
    """One system's standing under each instrument, over the scored item set."""

    system_id: str
    oracle_mean: float
    judge_mean: float
    judge_mean_pp: float
    oracle_rank: int
    judge_rank: int


class OracleReport(TypedDict):
    """The ADR-029 payload for one environment."""

    env_id: str
    task: str
    scale: str
    n_items: int
    facets: List[str]
    systems: List[SystemRow]
    gaps: List[OracleGap]
    claim_rates: List[ClaimRate]
    #: ADR-029 §e. Empty on environments whose pairs the reference cannot tell
    #: apart at all — a curve needs a spread of gaps to be a curve.
    detection: List[DetectionPoint]
    #: One row per facet when a per-facet breakdown ran; empty otherwise.
    facet_reports: List[FacetReport]
    #: Two boundary rows per budget wherever ``detection`` is non-empty.
    detection_summary: List[BoundarySummary]
    oracle_spread_pp: float
    judge_spread_pp: float
    rank_concordance: Optional[float]
    n_separated_pairs: int


def load_summeval_oracle(
    inputs_dir: str,
    *,
    facets: Sequence[str] = SUMMEVAL_FACETS,
) -> Dict[str, Dict[str, float]]:
    """Expert annotations for the pinned SummEval snapshot. Reads only, never fetches."""
    payload = read_source_cache(inputs_dir, SUMMEVAL_SOURCE_SPEC)
    return summeval_human_scores(payload, facets=facets)


def _cell_means(
    records: Sequence[ScoreRecord],
    env_id: str,
    split: RepeatSplit,
) -> Tuple[Dict[rec.CellKey, Dict[int, float]], List[str], List[str]]:
    """Estimation-repeat cells for *env_id*, plus its scored item and system ids."""
    env_records = rec.filter_records(records, env_id=env_id)
    if not env_records:
        raise ValueError(f"no records for env_id={env_id!r}")
    rec.check_schema(env_records)
    cells = rec.group_by_cell(env_records, repeats=list(split["estimation"]))
    if not cells:
        raise ValueError(f"env_id={env_id!r}: no cell inside the estimation repeats")
    items = sorted({key.item_id for key in cells})
    systems = sorted({key.system_id for key in cells})
    return cells, items, systems


def _paired_differences(
    values: Dict[str, Dict[str, float]],
    items: Sequence[str],
    system_a: str,
    system_b: str,
) -> List[float]:
    """Per-item ``a - b`` differences, for items both systems are scored on."""
    out: List[float] = []
    for item_id in items:
        row = values.get(item_id)
        if row is None or system_a not in row or system_b not in row:
            continue
        out.append(row[system_a] - row[system_b])
    return out


def oracle_gaps(
    oracle: Dict[str, Dict[str, float]],
    judge: Dict[str, Dict[str, float]],
    items: Sequence[str],
    systems: Sequence[str],
    scale: str,
    *,
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> List[OracleGap]:
    """Every scored pair's external gap, its interval, and the judge's residual.

    Both gaps are reported in %p of their own instrument's scale (ADR-001), which
    is what makes them comparable: the oracle's facets run 1--5 (width 4) and the
    judge's scale has whatever width it has. The interval is a seeded bootstrap
    over items — the unit the paired difference is averaged over — rather than a
    t-interval, so no normality is assumed for a 100-item mean of ordinal scores.
    """
    out: List[OracleGap] = []
    for index, (system_a, system_b) in enumerate(itertools.combinations(sorted(systems), 2)):
        oracle_diffs = _paired_differences(oracle, items, system_a, system_b)
        judge_diffs = _paired_differences(judge, items, system_a, system_b)
        if not oracle_diffs or not judge_diffs:
            continue
        oracle_pp = [value / SUMMEVAL_HUMAN_WIDTH * 100.0 for value in oracle_diffs]

        def statistic(sample: Sequence[float]) -> Sequence[Optional[float]]:
            return [statistics.fmean(sample) if sample else None]

        boot = bootstrap_vector(
            oracle_pp,
            statistic,
            seed=seed + ORACLE_SEED_OFFSET + 31 * index,
            n_resamples=n_resamples,
            ci_level=ci_level,
        )[0]
        gap_pp = statistics.fmean(oracle_pp)
        judge_gap_pp = to_pp(statistics.fmean(judge_diffs), scale)
        t_stat: Optional[float] = None
        if len(oracle_pp) > 1:
            spread = statistics.stdev(oracle_pp)
            if spread > 0.0:
                t_stat = gap_pp / (spread / math.sqrt(len(oracle_pp)))
        ci_lo, ci_hi = boot["ci_lo"], boot["ci_hi"]
        separated = ci_lo is not None and ci_hi is not None and (ci_lo > 0.0 or ci_hi < 0.0)
        out.append(
            OracleGap(
                system_a=system_a,
                system_b=system_b,
                n_items=len(oracle_pp),
                oracle_gap_pp=gap_pp,
                oracle_ci_lo_pp=ci_lo,
                oracle_ci_hi_pp=ci_hi,
                oracle_t=t_stat,
                separated=separated,
                judge_gap_pp=judge_gap_pp,
                residual_pp=judge_gap_pp - gap_pp,
                facets_separating=[],
                facet_gaps_pp={},
                bootstrap=boot["meta"],
            )
        )
    return out


def facet_breakdown(
    gaps: Sequence[OracleGap],
    oracle_by_facet: Dict[str, Dict[str, Dict[str, float]]],
    judge: Dict[str, Dict[str, float]],
    items: Sequence[str],
    systems: Sequence[str],
    scale: str,
    *,
    seed: int = DEFAULT_SEED,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> List[FacetReport]:
    """Re-run the gap analysis per single facet and annotate *gaps* in place.

    Each facet goes through :func:`oracle_gaps` with the SAME seed as the main
    run, so a facet row is exactly what the report would say had that facet been
    the whole reference — nothing about the per-facet numbers depends on a
    second seed path. Two products: per-pair ``facets_separating`` /
    ``facet_gaps_pp`` written onto *gaps*, and one :class:`FacetReport` per
    facet. Mutating in place keeps every pair's facet answer next to its
    average answer, which is where the aggregation-artifact check reads it.
    """
    by_pair = {(gap["system_a"], gap["system_b"]): gap for gap in gaps}
    reports: List[FacetReport] = []
    for facet in sorted(oracle_by_facet):
        oracle = oracle_by_facet[facet]
        facet_gaps = oracle_gaps(
            oracle,
            judge,
            items,
            systems,
            scale,
            seed=seed,
            n_resamples=n_resamples,
            ci_level=ci_level,
        )
        for facet_gap in facet_gaps:
            target = by_pair.get((facet_gap["system_a"], facet_gap["system_b"]))
            if target is None:
                continue
            target["facet_gaps_pp"][facet] = facet_gap["oracle_gap_pp"]
            if facet_gap["separated"]:
                target["facets_separating"].append(facet)
        means = {
            system_id: statistics.fmean(
                oracle[item_id][system_id]
                for item_id in items
                if system_id in oracle.get(item_id, {})
            )
            for system_id in systems
        }
        reports.append(
            FacetReport(
                facet=facet,
                n_separated_pairs=sum(1 for gap in facet_gaps if gap["separated"]),
                rank_concordance=rank_concordance(facet_gaps),
                oracle_spread_pp=(
                    (max(means.values()) - min(means.values())) / SUMMEVAL_HUMAN_WIDTH * 100.0
                ),
            )
        )
    for gap in gaps:
        gap["facets_separating"].sort()
    return reports


def detection_summaries(
    gaps: Sequence[OracleGap],
    detection: Sequence[DetectionPoint],
) -> List[BoundarySummary]:
    """Mean detection rate on either side of the two boundaries, per budget.

    Pure arithmetic over already-recorded rows — no draws, no randomness — kept
    in the payload so the manuscript's boundary numbers resolve to a stored
    field rather than to a reduction a reader must reproduce (AGENTS.md §3.6).
    """
    judge_gap = {(gap["system_a"], gap["system_b"]): abs(gap["judge_gap_pp"]) for gap in gaps}
    out: List[BoundarySummary] = []
    for n_repeats in sorted({point["n_repeats"] for point in detection}):
        rows = [point for point in detection if point["n_repeats"] == n_repeats]
        mdi_pp = rows[0]["mdi_pp"]
        for boundary, value, key in (
            ("judge_gap_vs_mdi", mdi_pp, lambda p: judge_gap[(p["system_a"], p["system_b"])]),
            ("oracle_gap_pp", EXTERNAL_BOUNDARY_PP, lambda p: abs(p["oracle_gap_pp"])),
        ):
            below = [p["detect_rate"] for p in rows if key(p) < value]
            above = [p["detect_rate"] for p in rows if key(p) >= value]
            out.append(
                BoundarySummary(
                    n_repeats=n_repeats,
                    boundary=boundary,
                    boundary_value_pp=value,
                    n_below=len(below),
                    n_above=len(above),
                    mean_detect_below=statistics.fmean(below) if below else None,
                    mean_detect_above=statistics.fmean(above) if above else None,
                )
            )
    return out


def bias_decomposition(gaps: Sequence[OracleGap]) -> Dict[str, float]:
    """Summarize ``judge gap = oracle gap + residual`` over the scored pairs.

    The residual is not "judge error" in any absolute sense — both terms are
    measurements — but it is the part of the observed ordering that survives
    subtracting everything the external reference can account for, and repeats do
    not shrink it.
    """
    if not gaps:
        return {}
    oracle_abs = [abs(gap["oracle_gap_pp"]) for gap in gaps]
    judge_abs = [abs(gap["judge_gap_pp"]) for gap in gaps]
    residual_abs = [abs(gap["residual_pp"]) for gap in gaps]
    return {
        "mean_abs_oracle_gap_pp": statistics.fmean(oracle_abs),
        "mean_abs_judge_gap_pp": statistics.fmean(judge_abs),
        "mean_abs_residual_pp": statistics.fmean(residual_abs),
        "max_abs_oracle_gap_pp": max(oracle_abs),
        "max_abs_judge_gap_pp": max(judge_abs),
        "residual_share": (
            statistics.fmean(residual_abs) / statistics.fmean(judge_abs)
            if statistics.fmean(judge_abs) > 0.0
            else 0.0
        ),
    }


def _pair_claim_rate(
    cells: Dict[rec.CellKey, Dict[int, float]],
    env_id: str,
    system_a: str,
    system_b: str,
    items: Sequence[str],
    n_repeats: int,
    threshold: float,
    draws: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """(claims, draws) for ``|mean_a - mean_b| > threshold`` under repeated scoring."""
    claims = 0
    made = 0
    usable = [
        item_id
        for item_id in items
        if rec.CellKey(env_id, item_id, system_a) in cells
        and rec.CellKey(env_id, item_id, system_b) in cells
    ]
    if not usable:
        return 0, 0
    for _ in range(draws):
        total = 0.0
        for item_id in usable:
            pool_a = rec.cell_values(cells[rec.CellKey(env_id, item_id, system_a)])
            pool_b = rec.cell_values(cells[rec.CellKey(env_id, item_id, system_b)])
            total += draw_mean(pool_a, n_repeats, rng) - draw_mean(pool_b, n_repeats, rng)
        made += 1
        if abs(total / len(usable)) > threshold:
            claims += 1
    return claims, made


def claim_rates(
    records: Sequence[ScoreRecord],
    env_id: str,
    *,
    split: RepeatSplit,
    null_pairs: Sequence[Tuple[str, str]],
    thresholds: Dict[int, float],
    alpha: float,
    scale: str,
    seed: int = DEFAULT_SEED,
    draws_per_pair: int = DEFAULT_DRAWS_PER_PAIR,
) -> List[ClaimRate]:
    """Claim rate at each budget on internally- and externally-null pairs (ADR-029b).

    *thresholds* maps a budget to the MDI in raw score units for that budget.
    The internal rate re-draws two scorings of one system (the estimator's own
    null, so it should land near *alpha*); the external rate draws the two
    different systems of each externally-null pair. Both use the same machinery,
    so any difference between them is the systems, not the sampling.
    """
    validate_split(split)
    cells, items, systems = _cell_means(records, env_id, split)
    out: List[ClaimRate] = []
    for n_repeats in sorted(thresholds):
        threshold = thresholds[n_repeats]
        rng = make_rng(seed + ORACLE_SEED_OFFSET + 17 * n_repeats)
        external_claims = external_draws = 0
        for system_a, system_b in sorted(null_pairs):
            claims, made = _pair_claim_rate(
                cells, env_id, system_a, system_b, items, n_repeats, threshold, draws_per_pair, rng
            )
            external_claims += claims
            external_draws += made
        internal_claims = internal_draws = 0
        for system_id in systems:
            claims, made = _pair_claim_rate(
                cells,
                env_id,
                system_id,
                system_id,
                items,
                n_repeats,
                threshold,
                draws_per_pair,
                rng,
            )
            internal_claims += claims
            internal_draws += made
        if not external_draws or not internal_draws:
            continue
        out.append(
            ClaimRate(
                n_repeats=n_repeats,
                alpha=alpha,
                mdi_pp=to_pp(threshold, scale),
                n_null_pairs=len(null_pairs),
                n_draws=external_draws,
                internal_rate=internal_claims / internal_draws,
                external_rate=external_claims / external_draws,
            )
        )
    return out


def detection_curve(
    records: Sequence[ScoreRecord],
    env_id: str,
    *,
    split: RepeatSplit,
    gaps: Sequence[OracleGap],
    thresholds: Dict[int, float],
    scale: str,
    seed: int = DEFAULT_SEED,
    draws_per_pair: int = DEFAULT_DRAWS_PER_PAIR,
) -> List[DetectionPoint]:
    """Per-pair detection rate at each budget, against the reference's gap (ADR-029 §e).

    Same draw machinery as :func:`claim_rates` — so a zero-gap row here and the
    external rate there are the same computation — but reported per pair instead
    of pooled, which is what turns it into a curve over effect size.

    Rows are emitted in sorted (pair, budget) order and the per-budget seed offset
    matches :func:`claim_rates`, so a pair that appears in both carries the same
    draws in both. Pairs the environment did not score are skipped rather than
    reported as zero.
    """
    validate_split(split)
    cells, items, _systems = _cell_means(records, env_id, split)
    out: List[DetectionPoint] = []
    for n_repeats in sorted(thresholds):
        threshold = thresholds[n_repeats]
        rng = make_rng(seed + ORACLE_SEED_OFFSET + 17 * n_repeats)
        for gap in sorted(gaps, key=lambda row: (row["system_a"], row["system_b"])):
            claims, made = _pair_claim_rate(
                cells,
                env_id,
                gap["system_a"],
                gap["system_b"],
                items,
                n_repeats,
                threshold,
                draws_per_pair,
                rng,
            )
            if not made:
                continue
            out.append(
                DetectionPoint(
                    system_a=gap["system_a"],
                    system_b=gap["system_b"],
                    oracle_gap_pp=gap["oracle_gap_pp"],
                    separated=gap["separated"],
                    n_repeats=n_repeats,
                    mdi_pp=to_pp(threshold, scale),
                    n_draws=made,
                    detect_rate=claims / made,
                )
            )
    return out


def _rank_map(values: Dict[str, float]) -> Dict[str, int]:
    """Dense rank, best first; ties share the lower rank."""
    ordered = sorted(values, key=lambda key: (-values[key], key))
    ranks: Dict[str, int] = {}
    position = 0
    previous: Optional[float] = None
    for index, key in enumerate(ordered):
        if previous is None or values[key] != previous:
            position = index + 1
            previous = values[key]
        ranks[key] = position
    return ranks


def rank_concordance(gaps: Sequence[OracleGap]) -> Optional[float]:
    """Share of scored pairs the two instruments order the same way.

    Reported instead of a rank correlation because the pair count here is small
    enough (six for four systems) that a correlation coefficient would carry a
    precision the data does not have. Pairs the oracle scores as an exact tie are
    excluded: there is no ordering to agree with, and if that leaves nothing the
    answer is ``None`` rather than a NaN — the derived tree has to stay valid
    JSON, and ``NaN`` is not (AGENTS.md §3.3).
    """
    comparable = [gap for gap in gaps if gap["oracle_gap_pp"] != 0.0]
    if not comparable:
        return None
    agree = sum(
        1
        for gap in comparable
        if (gap["oracle_gap_pp"] > 0.0) == (gap["judge_gap_pp"] > 0.0)
        and gap["judge_gap_pp"] != 0.0
    )
    return agree / len(comparable)


def build_oracle_report(
    records: Iterable[ScoreRecord],
    env_id: str,
    oracle: Dict[str, Dict[str, float]],
    *,
    split: RepeatSplit,
    thresholds: Dict[int, float],
    alpha: float,
    facets: Sequence[str],
    oracle_by_facet: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    seed: int = DEFAULT_SEED,
    draws_per_pair: int = DEFAULT_DRAWS_PER_PAIR,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> OracleReport:
    """The full ADR-029 payload for one environment.

    *oracle_by_facet* maps each facet name to its own single-facet annotation
    dict; when it carries two or more facets the report gains the per-facet
    breakdown (ADR-033 construct check). With one facet the breakdown would
    restate the average, so it is skipped as trivially true.
    """
    ordered = rec.sorted_records(records)
    env_records = rec.filter_records(ordered, env_id=env_id)
    if not env_records:
        raise ValueError(f"no records for env_id={env_id!r}")
    meta = {m["env_id"]: m for m in rec.env_metadata(env_records)}[env_id]
    scale = meta["scale"]
    cells, items, systems = _cell_means(env_records, env_id, split)

    judge: Dict[str, Dict[str, float]] = {}
    for key, cell in cells.items():
        values = rec.cell_values(cell)
        if values:
            judge.setdefault(key.item_id, {})[key.system_id] = statistics.fmean(values)

    missing = [item_id for item_id in items if item_id not in oracle]
    if missing:
        raise ValueError(
            f"env_id={env_id!r}: {len(missing)} scored item(s) carry no oracle annotation, "
            f"first={missing[0]!r} — the snapshot and the store disagree on the item set"
        )

    gaps = oracle_gaps(
        oracle,
        judge,
        items,
        systems,
        scale,
        seed=seed,
        n_resamples=n_resamples,
        ci_level=ci_level,
    )
    null_pairs = [(gap["system_a"], gap["system_b"]) for gap in gaps if not gap["separated"]]
    rates = claim_rates(
        env_records,
        env_id,
        split=split,
        null_pairs=null_pairs,
        thresholds=thresholds,
        alpha=alpha,
        scale=scale,
        seed=seed,
        draws_per_pair=draws_per_pair,
    )

    # A curve needs a spread of gaps. Where the reference separates nothing
    # (the close-pair environments of §a-§c) every rung would sit at the same x,
    # and `claim_rates` above already reports that pooled — so the field stays
    # empty there rather than restating it one pair at a time.
    detection = (
        detection_curve(
            env_records,
            env_id,
            split=split,
            gaps=gaps,
            thresholds=thresholds,
            scale=scale,
            seed=seed,
            draws_per_pair=draws_per_pair,
        )
        if any(gap["separated"] for gap in gaps)
        else []
    )

    facet_rows = (
        facet_breakdown(
            gaps,
            oracle_by_facet,
            judge,
            items,
            systems,
            scale,
            seed=seed,
            n_resamples=n_resamples,
            ci_level=ci_level,
        )
        if oracle_by_facet is not None and len(oracle_by_facet) >= 2
        else []
    )
    summaries = detection_summaries(gaps, detection) if detection else []

    oracle_means = {
        system_id: statistics.fmean(
            oracle[item_id][system_id] for item_id in items if system_id in oracle[item_id]
        )
        for system_id in systems
    }
    judge_means = {
        system_id: statistics.fmean(
            judge[item_id][system_id] for item_id in items if system_id in judge.get(item_id, {})
        )
        for system_id in systems
    }
    oracle_rank = _rank_map(oracle_means)
    judge_rank = _rank_map(judge_means)
    rows = [
        SystemRow(
            system_id=system_id,
            oracle_mean=oracle_means[system_id],
            judge_mean=judge_means[system_id],
            judge_mean_pp=to_pp(judge_means[system_id], scale),
            oracle_rank=oracle_rank[system_id],
            judge_rank=judge_rank[system_id],
        )
        for system_id in sorted(systems)
    ]
    oracle_spread = (
        (max(oracle_means.values()) - min(oracle_means.values())) / SUMMEVAL_HUMAN_WIDTH * 100.0
    )
    judge_spread = to_pp(max(judge_means.values()) - min(judge_means.values()), scale)
    return OracleReport(
        env_id=env_id,
        task=meta["task"],
        scale=scale,
        n_items=len(items),
        facets=sorted(facets),
        systems=rows,
        gaps=gaps,
        claim_rates=rates,
        detection=detection,
        facet_reports=facet_rows,
        detection_summary=summaries,
        oracle_spread_pp=oracle_spread,
        judge_spread_pp=judge_spread,
        rank_concordance=rank_concordance(gaps),
        n_separated_pairs=sum(1 for gap in gaps if gap["separated"]),
    )
