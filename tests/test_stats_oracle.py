"""Tests for the external-oracle validation path (ADR-029)."""

import math
import statistics
from typing import Any, Dict, List

import pytest

from mdi.stats.fip import make_split
from mdi.stats.oracle import (
    EXTERNAL_BOUNDARY_PP,
    bias_decomposition,
    build_oracle_report,
    claim_rates,
    detection_curve,
    detection_summaries,
    facet_breakdown,
    oracle_gaps,
    rank_concordance,
)
from mdi.stats.synthetic import synthetic_records
from mdi.store import SUMMEVAL_FACETS, SUMMEVAL_HUMAN_WIDTH, summeval_human_scores

ENV = "e_synthetic"


def _payload(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """A SummEval datasets-server payload shaped like the pinned snapshot."""
    return {"rows": [{"row": row} for row in rows]}


def _row(item_id: str, per_system: List[float], n_facets: int = 4) -> Dict[str, Any]:
    """One SummEval row whose four facets all carry *per_system*."""
    row: Dict[str, Any] = {
        "id": item_id,
        "text": "source",
        "machine_summaries": [f"summary {index}" for index in range(len(per_system))],
    }
    for facet in list(SUMMEVAL_FACETS)[:n_facets]:
        row[facet] = list(per_system)
    return row


# --- the annotation reader --------------------------------------------------------------


def test_human_scores_average_the_facets_and_key_on_the_scored_system_ids() -> None:
    """Facets are averaged with equal weight and systems keyed as s_<index>."""
    # Given: a row whose facets disagree — relevance and coherence are moved off
    # the [1, 5] baseline the other two keep
    row = _row("i_0", [1.0, 5.0])
    row["coherence"] = [1.0, 3.0]
    row["relevance"] = [3.0, 5.0]
    # When: the annotations are read
    scores = summeval_human_scores(_payload([row]))
    # Then: the ids match the scored records and each score is the facet mean
    assert sorted(scores["i_0"]) == ["s_00", "s_01"]
    # s_00: relevance 3, coherence 1, fluency 1, consistency 1
    assert scores["i_0"]["s_00"] == pytest.approx((3.0 + 1.0 + 1.0 + 1.0) / 4)
    # s_01: relevance 5, coherence 3, fluency 5, consistency 5
    assert scores["i_0"]["s_01"] == pytest.approx((5.0 + 3.0 + 5.0 + 5.0) / 4)


def test_human_scores_reject_a_payload_missing_an_annotation_facet() -> None:
    """A payload without the requested facet is an error, not a silent zero."""
    # Given: a row carrying only one facet
    row = _row("i_0", [3.0, 4.0], n_facets=1)
    # When / Then: asking for all four names the missing ones
    with pytest.raises(ValueError, match="annotation facet"):
        summeval_human_scores(_payload([row]))


# --- gaps and the externally-null set ----------------------------------------------------


def test_oracle_gap_is_exactly_zero_for_two_equally_rated_systems() -> None:
    """The non-circular null: a pair the reference scores identically, item by item."""
    # Given: two systems the oracle rates the same on every item, and a judge
    # that nonetheless separates them
    oracle = {f"i_{i}": {"s_00": 4.0 + 0.1 * i, "s_01": 4.0 + 0.1 * i} for i in range(20)}
    judge = {f"i_{i}": {"s_00": 3.0, "s_01": 2.0} for i in range(20)}
    # When: the gaps are computed
    gaps = oracle_gaps(oracle, judge, list(oracle), ["s_00", "s_01"], "likert5", n_resamples=200)
    # Then: the external gap is zero, the pair is not separated ...
    assert len(gaps) == 1
    assert gaps[0]["oracle_gap_pp"] == pytest.approx(0.0)
    assert gaps[0]["separated"] is False
    # ... and the whole judge gap lands in the residual
    assert gaps[0]["judge_gap_pp"] == pytest.approx(25.0)
    assert gaps[0]["residual_pp"] == pytest.approx(25.0)


def test_oracle_marks_a_pair_separated_when_its_interval_excludes_zero() -> None:
    """A pair the reference *can* resolve is excluded from the null set."""
    # Given: a consistent one-point gap on the 1-5 facet scale
    oracle = {f"i_{i}": {"s_00": 4.0, "s_01": 3.0} for i in range(30)}
    judge = {f"i_{i}": {"s_00": 3.0, "s_01": 3.0} for i in range(30)}
    # When: the gaps are computed
    gaps = oracle_gaps(oracle, judge, list(oracle), ["s_00", "s_01"], "likert5", n_resamples=200)
    # Then: it is separated, at 1/4 of the scale width
    assert gaps[0]["separated"] is True
    assert gaps[0]["oracle_gap_pp"] == pytest.approx(100.0 / SUMMEVAL_HUMAN_WIDTH)


def test_rank_concordance_counts_pairs_the_two_instruments_order_alike() -> None:
    """Agreement is a share of orderable pairs; oracle ties are not orderable."""
    # Given: three pairs — one agreeing, one disagreeing, one an oracle tie
    gaps = [
        {"oracle_gap_pp": 1.0, "judge_gap_pp": 2.0},
        {"oracle_gap_pp": 1.0, "judge_gap_pp": -2.0},
        {"oracle_gap_pp": 0.0, "judge_gap_pp": 5.0},
    ]
    # When / Then: the tie is dropped and one of the remaining two agrees
    assert rank_concordance(gaps) == pytest.approx(0.5)  # type: ignore[arg-type]


def test_bias_decomposition_attributes_the_whole_gap_when_the_oracle_is_flat() -> None:
    """With no external difference the residual carries the entire observed gap."""
    # Given: gaps whose oracle side is zero
    gaps = [
        {"oracle_gap_pp": 0.0, "judge_gap_pp": 4.0, "residual_pp": 4.0},
        {"oracle_gap_pp": 0.0, "judge_gap_pp": -2.0, "residual_pp": -2.0},
    ]
    # When: the decomposition is summarized
    summary = bias_decomposition(gaps)  # type: ignore[arg-type]
    # Then: the residual share is one
    assert summary["mean_abs_oracle_gap_pp"] == pytest.approx(0.0)
    assert summary["residual_share"] == pytest.approx(1.0)


# --- claim rates -------------------------------------------------------------------------


def test_internal_claim_rate_tracks_the_threshold_and_external_tracks_the_bias() -> None:
    """The threshold controls re-scoring the same system; it says nothing about two systems."""
    # Given: two systems separated by a large constant the judge always sees
    records = synthetic_records(
        system_means={"s_00": 0.0, "s_01": 1.0},
        n_items=20,
        n_repeats=10,
        sigma=0.3,
        seed=17,
    )
    split = make_split(screening=[], estimation=list(range(10)))
    # When: claim rates are measured against a threshold far below that constant
    rates = claim_rates(
        records,
        ENV,
        split=split,
        null_pairs=[("s_00", "s_01")],
        thresholds={1: 0.15, 5: 0.10},
        alpha=0.05,
        scale="likert5",
        draws_per_pair=200,
    )
    # Then: re-scoring one system rarely clears it ...
    assert rates
    for rate in rates:
        assert rate["internal_rate"] < 0.2
        # ... while the two systems clear it essentially always, and more so with
        # budget: repeats buy confidence in the ordering, not correctness
        assert rate["external_rate"] > 0.9
    assert rates[-1]["external_rate"] >= rates[0]["external_rate"]


# --- the known-effect detection curve (ADR-029 §e) ---------------------------------------


def _ladder_records(seed: int = 23) -> Any:
    """Three systems at 0.0 / 0.0 / 1.0 — one zero-gap rung and two wide ones."""
    return synthetic_records(
        system_means={"s_00": 0.0, "s_01": 0.0, "s_02": 1.0},
        n_items=20,
        n_repeats=10,
        sigma=0.3,
        seed=seed,
    )


def _ladder_gaps(records: Any) -> Any:
    """Oracle gaps for :func:`_ladder_records`, annotations tracking the true means."""
    # s_00 and s_01 are annotated identically; s_02 sits a full point above both.
    oracle = {record["item_id"]: {"s_00": 4.0, "s_01": 4.0, "s_02": 5.0} for record in records}
    pools: Dict[str, Dict[str, List[float]]] = {}
    for record in records:
        pools.setdefault(record["item_id"], {}).setdefault(record["system_id"], []).append(
            float(record["parsed_score"])
        )
    judge = {
        item_id: {system_id: statistics.fmean(values) for system_id, values in row.items()}
        for item_id, row in pools.items()
    }
    items = sorted(oracle)
    return oracle_gaps(oracle, judge, items, ["s_00", "s_01", "s_02"], "likert5", seed=5)


def test_detection_rate_rises_with_the_external_gap() -> None:
    """The curve must separate a zero-gap rung from a wide one at the same budget."""
    # Given: three systems, two identical and one a full point above them
    records = _ladder_records()
    split = make_split(screening=[], estimation=list(range(10)))
    gaps = _ladder_gaps(records)
    # When: every pair is measured against the same threshold at one budget
    points = detection_curve(
        records,
        ENV,
        split=split,
        gaps=gaps,
        thresholds={5: 0.15},
        scale="likert5",
        draws_per_pair=200,
    )
    # Then: the zero-gap pair rarely fires and the wide pairs essentially always do
    by_pair = {(p["system_a"], p["system_b"]): p["detect_rate"] for p in points}
    assert by_pair[("s_00", "s_01")] < 0.5
    assert by_pair[("s_00", "s_02")] > 0.9
    assert by_pair[("s_01", "s_02")] > 0.9


def test_detection_curve_carries_the_gap_and_the_budget_on_every_row() -> None:
    """Each row must be readable as a point (x=gap, y=rate) without a second lookup."""
    # Given: the ladder records and two budgets
    records = _ladder_records()
    split = make_split(screening=[], estimation=list(range(10)))
    gaps = _ladder_gaps(records)
    # When: the curve is built over two budgets
    points = detection_curve(
        records,
        ENV,
        split=split,
        gaps=gaps,
        thresholds={1: 0.2, 5: 0.15},
        scale="likert5",
        draws_per_pair=50,
    )
    # Then: one row per (pair, budget), each self-describing — the gap it carries
    # is the signed one from the matching OracleGap, so no second lookup is needed
    assert len(points) == len(gaps) * 2
    assert {p["n_repeats"] for p in points} == {1, 5}
    gap_by_pair = {(g["system_a"], g["system_b"]): g for g in gaps}
    for point in points:
        assert 0.0 <= point["detect_rate"] <= 1.0
        assert point["n_draws"] == 50
        source = gap_by_pair[(point["system_a"], point["system_b"])]
        assert point["oracle_gap_pp"] == source["oracle_gap_pp"]
        assert point["separated"] == source["separated"]


def test_detection_curve_is_deterministic_under_a_fixed_seed() -> None:
    """Repeat calls must agree exactly — the curve lands in derived output."""
    # Given: identical inputs
    records = _ladder_records()
    split = make_split(screening=[], estimation=list(range(10)))
    gaps = _ladder_gaps(records)
    kwargs: Dict[str, Any] = {
        "split": split,
        "gaps": gaps,
        "thresholds": {3: 0.15},
        "scale": "likert5",
        "draws_per_pair": 40,
    }
    # When: the curve is built twice
    first = detection_curve(records, ENV, **kwargs)
    second = detection_curve(records, ENV, **kwargs)
    # Then: bit for bit (AGENTS.md §3.3)
    assert first == second


def test_report_omits_the_curve_when_the_reference_separates_nothing() -> None:
    """Close-pair environments get an empty field, not a flat line restating claim_rates."""
    # Given: two systems the oracle rates identically
    records = synthetic_records(
        system_means={"s_00": 0.0, "s_01": 0.1}, n_items=8, n_repeats=6, sigma=0.4, seed=11
    )
    split = make_split(screening=[], estimation=list(range(6)))
    oracle = {record["item_id"]: {"s_00": 4.0, "s_01": 4.0} for record in records}
    # When: the report is assembled
    report = build_oracle_report(
        records,
        ENV,
        oracle,
        split=split,
        thresholds={1: 0.2},
        alpha=0.05,
        facets=SUMMEVAL_FACETS,
    )
    # Then: nothing is separated, so the curve stays empty and the pooled rates carry it
    assert report["n_separated_pairs"] == 0
    assert report["detection"] == []
    assert report["claim_rates"]


# --- the per-facet construct check (ADR-033) ---------------------------------------------


def test_facet_breakdown_annotates_gaps_and_finds_no_artifact_when_one_facet_carries_it() -> None:
    """A pair separated on the average must name the facet(s) that support it."""
    # Given: two systems a full point apart on coherence, identical on fluency,
    # so the two-facet average still separates them
    items = [f"i_{i}" for i in range(30)]
    coherence = {i: {"s_00": 4.0, "s_01": 3.0} for i in items}
    fluency = {i: {"s_00": 4.0, "s_01": 4.0} for i in items}
    average = {i: {"s_00": 4.0, "s_01": 3.5} for i in items}
    judge = {i: {"s_00": 3.0, "s_01": 3.0} for i in items}
    gaps = oracle_gaps(average, judge, items, ["s_00", "s_01"], "likert5", n_resamples=200)
    assert gaps[0]["separated"] is True
    # When: the breakdown runs over the two facets
    reports = facet_breakdown(
        gaps,
        {"coherence": coherence, "fluency": fluency},
        judge,
        items,
        ["s_00", "s_01"],
        "likert5",
        n_resamples=200,
    )
    # Then: the separating facet is named on the pair, its gap is carried, and
    # the separation is supported by a facet — not an aggregation artifact
    assert gaps[0]["facets_separating"] == ["coherence"]
    assert set(gaps[0]["facet_gaps_pp"]) == {"coherence", "fluency"}
    assert gaps[0]["facet_gaps_pp"]["fluency"] == pytest.approx(0.0)
    by_facet = {row["facet"]: row for row in reports}
    assert by_facet["coherence"]["n_separated_pairs"] == 1
    assert by_facet["fluency"]["n_separated_pairs"] == 0
    assert by_facet["coherence"]["oracle_spread_pp"] == pytest.approx(25.0)


def test_report_carries_empty_facet_fields_without_a_per_facet_oracle() -> None:
    """No breakdown input, no breakdown output — and the gap fields stay empty, not absent."""
    # Given: a report built the pre-breakdown way
    records = synthetic_records(
        system_means={"s_00": 0.0, "s_01": 0.3}, n_items=8, n_repeats=6, sigma=0.4, seed=9
    )
    split = make_split(screening=[], estimation=list(range(6)))
    oracle = {record["item_id"]: {"s_00": 4.0, "s_01": 4.0} for record in records}
    # When: no oracle_by_facet is supplied
    report = build_oracle_report(
        records,
        ENV,
        oracle,
        split=split,
        thresholds={1: 0.2},
        alpha=0.05,
        facets=SUMMEVAL_FACETS,
    )
    # Then: the fields exist and are empty
    assert report["facet_reports"] == []
    for gap in report["gaps"]:
        assert gap["facets_separating"] == []
        assert gap["facet_gaps_pp"] == {}


def test_detection_summary_reports_both_boundaries_with_their_values() -> None:
    """Each budget must carry a design-confirmation row and an external-boundary row."""
    # Given: the ladder records and their gaps
    records = _ladder_records()
    split = make_split(screening=[], estimation=list(range(10)))
    gaps = _ladder_gaps(records)
    points = detection_curve(
        records,
        ENV,
        split=split,
        gaps=gaps,
        thresholds={1: 0.2, 5: 0.15},
        scale="likert5",
        draws_per_pair=50,
    )
    # When: the summaries are computed
    summaries = detection_summaries(gaps, points)
    # Then: two boundaries per budget, each self-describing and exhaustive
    assert len(summaries) == 4
    by_key = {(s["n_repeats"], s["boundary"]): s for s in summaries}
    assert by_key[(1, "judge_gap_vs_mdi")]["boundary_value_pp"] == pytest.approx(0.2 / 4.0 * 100.0)
    assert by_key[(1, "oracle_gap_pp")]["boundary_value_pp"] == EXTERNAL_BOUNDARY_PP
    for summary in summaries:
        assert summary["n_below"] + summary["n_above"] == len(gaps)
        for side in ("mean_detect_below", "mean_detect_above"):
            value = summary[side]  # type: ignore[literal-required,unused-ignore]
            assert value is None or 0.0 <= value <= 1.0


def test_facet_breakdown_is_deterministic_and_matches_a_single_facet_run() -> None:
    """A facet row must equal what the report would say were that facet the whole oracle."""
    # Given: ladder-shaped records and two facets that disagree
    items = [f"i_{i}" for i in range(25)]
    facet_a = {i: {"s_00": 4.0, "s_01": 3.0} for i in items}
    facet_b = {i: {"s_00": 4.0, "s_01": 4.2} for i in items}
    average = {i: {"s_00": 4.0, "s_01": 3.6} for i in items}
    judge = {i: {"s_00": 3.0, "s_01": 2.5} for i in items}
    systems = ["s_00", "s_01"]
    # When: the breakdown runs twice, and the facet is also run alone
    first_gaps = oracle_gaps(average, judge, items, systems, "likert5", n_resamples=150)
    first = facet_breakdown(
        first_gaps,
        {"a": facet_a, "b": facet_b},
        judge,
        items,
        systems,
        "likert5",
        n_resamples=150,
    )
    second_gaps = oracle_gaps(average, judge, items, systems, "likert5", n_resamples=150)
    second = facet_breakdown(
        second_gaps,
        {"a": facet_a, "b": facet_b},
        judge,
        items,
        systems,
        "likert5",
        n_resamples=150,
    )
    alone = oracle_gaps(facet_a, judge, items, systems, "likert5", n_resamples=150)
    # Then: bit-for-bit repeatable, and the facet row equals the standalone run
    assert first == second and first_gaps == second_gaps
    assert first[0]["facet"] == "a"
    assert first[0]["n_separated_pairs"] == sum(1 for gap in alone if gap["separated"])
    assert first_gaps[0]["facet_gaps_pp"]["a"] == alone[0]["oracle_gap_pp"]


# --- the assembled report ----------------------------------------------------------------


def test_oracle_report_refuses_an_item_set_the_snapshot_does_not_cover() -> None:
    """A store and a snapshot that disagree on items is an error, not a partial answer."""
    # Given: records over items the oracle has no annotation for
    records = synthetic_records(
        system_means={"s_00": 0.0, "s_01": 0.2}, n_items=4, n_repeats=4, sigma=0.2, seed=5
    )
    split = make_split(screening=[], estimation=list(range(4)))
    # When / Then: the mismatch is named
    with pytest.raises(ValueError, match="no oracle annotation"):
        build_oracle_report(
            records,
            ENV,
            {"not_an_item": {"s_00": 4.0, "s_01": 4.0}},
            split=split,
            thresholds={1: 0.1},
            alpha=0.05,
            facets=SUMMEVAL_FACETS,
        )


def test_oracle_report_is_deterministic_under_a_fixed_seed() -> None:
    """Two runs of the whole report agree bit for bit (AGENTS.md §3.3)."""
    # Given: records and a matching oracle
    records = synthetic_records(
        system_means={"s_00": 0.0, "s_01": 0.3}, n_items=8, n_repeats=6, sigma=0.4, seed=9
    )
    split = make_split(screening=[], estimation=list(range(6)))
    oracle = {record["item_id"]: {"s_00": 4.0, "s_01": 4.0} for record in records}
    # When: the report is built twice
    reports = [
        build_oracle_report(
            records,
            ENV,
            oracle,
            split=split,
            thresholds={1: 0.2, 3: 0.1},
            alpha=0.05,
            facets=SUMMEVAL_FACETS,
            seed=23,
            draws_per_pair=50,
            n_resamples=100,
        )
        for _ in range(2)
    ]
    # Then: they are identical — which requires the undefined agreement to be
    # None rather than NaN, since NaN is neither equal to itself nor valid JSON
    assert reports[0] == reports[1]
    assert reports[0]["rank_concordance"] is None
    assert reports[0]["n_separated_pairs"] == 0
    assert math.isclose(reports[0]["oracle_spread_pp"], 0.0)
