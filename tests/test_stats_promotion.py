"""Tests for the Exp 4 promotion-reversal estimator (ADR-017)."""

import json
from typing import Dict, List, Sequence

import pytest

from mdi.stats.promotion import (
    mcnemar_exact,
    promotion_report,
    promotion_reversal_probability,
    prospective_min_delta,
    prospective_min_majority,
    stopping_counterfactual,
)
from mdi.store import PromotionRecord


def make_promo(
    seed: int, promo_idx: int, b: int, c: int, delta: float, n_questions: int = 100
) -> PromotionRecord:
    """Synthesize a promotion whose paired vectors realize exactly b gains and c losses."""
    if b + c > n_questions:
        raise ValueError("b + c cannot exceed n_questions")
    question_ids = [f"q{index:04d}" for index in range(n_questions)]
    base_correct: List[bool] = []
    transform_correct: List[bool] = []
    for index in range(n_questions):
        if index < b:  # wrong -> right (gain)
            base_correct.append(False)
            transform_correct.append(True)
        elif index < b + c:  # right -> wrong (loss)
            base_correct.append(True)
            transform_correct.append(False)
        else:  # concordant (right -> right)
            base_correct.append(True)
            transform_correct.append(True)
    return PromotionRecord(
        seed=seed,
        promo_idx=promo_idx,
        name="transform",
        iteration_id="loop-001",
        confirm_delta=delta,
        n_recovered=b,
        n_introduced=c,
        n_discordant=b + c,
        n_questions=n_questions,
        question_ids=question_ids,
        base_correct=base_correct,
        transform_correct=transform_correct,
    )


# seed 101's promotion sequence, straight from the pinned Regimes data (ADR-017).
SEED_101: Sequence[PromotionRecord] = [
    make_promo(101, 0, 7, 6, 0.01),
    make_promo(101, 1, 6, 1, 0.05),
    make_promo(101, 2, 9, 2, 0.07),
    make_promo(101, 3, 13, 4, 0.09),
    make_promo(101, 4, 5, 5, 0.00),
    make_promo(101, 5, 7, 6, 0.01),
]


def test_mcnemar_exact_matches_known_binomial_tails() -> None:
    """The two-sided exact McNemar p equals 2x the smaller exact binomial tail."""
    # Given / When / Then: hand-computable cases
    assert mcnemar_exact(7, 6) == pytest.approx(1.0)  # symmetric split: p = 1
    assert mcnemar_exact(11, 1) == pytest.approx(0.00634766, abs=1e-6)  # 26 / 4096
    assert mcnemar_exact(10, 0) == pytest.approx(0.00195313, abs=1e-6)  # 2 / 1024


def test_mcnemar_exact_is_symmetric_and_degenerate_at_zero() -> None:
    """The test does not care which direction the flips went; no pairs means no evidence."""
    # Given / When / Then
    assert mcnemar_exact(11, 1) == mcnemar_exact(1, 11)
    assert mcnemar_exact(0, 0) == 1.0


def test_reversal_probability_matches_the_one_sided_tail() -> None:
    """(7,6) is a coin toss (0.5); (11,1) almost never reproduces under noise."""
    # Given / When / Then
    assert promotion_reversal_probability(7, 6) == 0.5  # exact: 4096 / 8192
    assert promotion_reversal_probability(11, 1) == pytest.approx(0.00317383, abs=1e-6)


def test_reversal_probability_is_half_the_two_sided_mcnemar_off_the_boundary() -> None:
    """The two-sided McNemar p is exactly min(1, 2 x the one-sided reversal tail)."""
    # Given: a range of discordant splits
    for b, c in [(11, 1), (10, 0), (9, 2), (6, 1), (13, 4)]:
        # When / Then
        assert mcnemar_exact(b, c) == pytest.approx(
            min(1.0, 2.0 * promotion_reversal_probability(b, c))
        )


def test_reversal_probability_decreases_as_the_counts_separate() -> None:
    """For a fixed n, a more lopsided split is less consistent with noise (monotone down)."""
    # Given: splits of n = 13 growing steadily more lopsided
    splits = [(7, 6), (8, 5), (9, 4), (10, 3), (11, 2), (12, 1), (13, 0)]
    # When: their reversal probabilities are taken in order
    reversals = [promotion_reversal_probability(b, c) for b, c in splits]
    # Then: strictly decreasing
    assert all(earlier > later for earlier, later in zip(reversals, reversals[1:]))


def test_reversal_and_mcnemar_reject_negative_counts() -> None:
    """A negative discordant count is a caller bug, not a silently clamped input."""
    # Given / When / Then
    with pytest.raises(ValueError):
        promotion_reversal_probability(-1, 2)
    with pytest.raises(ValueError):
        mcnemar_exact(3, -1)


def test_prospective_min_majority_inverts_the_reversal_tail() -> None:
    """The stated-in-advance bar is exactly where the retrospective test starts passing."""
    # Given: the discordant counts the Regimes gates actually produced
    for n in (8, 10, 12, 13, 15, 17, 20):
        # When: the prospective minimum majority is computed at alpha = 0.05
        b = prospective_min_majority(n)
        assert b is not None
        # Then: b clears the band and b - 1 does not (tight inversion)
        assert promotion_reversal_probability(b, n - b) <= 0.05
        assert promotion_reversal_probability(b - 1, n - b + 1) > 0.05


def test_prospective_min_majority_known_anchors() -> None:
    """Hand-checked values: n=13 needs 10-vs-3; n=8 needs 7-vs-1."""
    # Given / When / Then
    assert prospective_min_majority(13) == 10
    assert prospective_min_majority(8) == 7


def test_prospective_min_majority_underpowered_gate_returns_none() -> None:
    """Too few discordant pairs -> no majority can clear the band (cf. seeds 11/23)."""
    # Given: n so small that even a unanimous split has P > alpha (n=4: 1/16 > 0.05... n=3: 1/8)
    # When / Then: 0..4 discordant pairs cannot license any promotion at alpha=0.05
    for n in (0, 1, 2, 3, 4):
        assert prospective_min_majority(n) is None
    assert prospective_min_majority(5) == 5  # 1/32 = 0.03125 <= 0.05


def test_prospective_min_delta_converts_to_accuracy_points() -> None:
    """On a 100-question gate, n=13 discordant needs at least +0.07 accuracy."""
    # Given / When
    delta = prospective_min_delta(13, 100)
    # Then
    assert delta == pytest.approx(0.07)
    assert prospective_min_delta(3, 100) is None


def test_prospective_functions_reject_bad_inputs() -> None:
    """Negative counts, non-positive question totals, and out-of-range alpha are caller bugs."""
    # Given / When / Then
    with pytest.raises(ValueError):
        prospective_min_majority(-1)
    with pytest.raises(ValueError):
        prospective_min_majority(10, alpha=0.0)
    with pytest.raises(ValueError):
        prospective_min_delta(10, 0)


def test_stopping_counterfactual_halts_seed_101_at_the_peak() -> None:
    """Our plateau rule stops seed 101 at promo #4 (13 vs 4, +0.09), the loop's peak."""
    # Given: seed 101's full promotion sequence
    # When: the stopping counterfactual is computed at alpha = 0.05
    result = stopping_counterfactual(SEED_101, alpha=0.05)
    # Then: it halts at the last band-clearing promotion, before the noise drift
    assert result["halt_promo_num"] == 4
    assert result["halt_promo_idx"] == 3
    assert result["halt_n_recovered"] == 13
    assert result["halt_n_introduced"] == 4
    assert result["peak_confirm_delta"] == pytest.approx(0.09)
    assert result["final_confirm_delta"] == pytest.approx(0.01)
    assert result["n_promotions_after_halt"] == 2
    assert result["drift_confirm_delta"] == pytest.approx(-0.08)


def test_stopping_counterfactual_is_empty_when_no_promotion_clears_the_band() -> None:
    """An underpowered seed (all splits within noise) should not have promoted at all."""
    # Given: seed 11's two underpowered promotions
    seed_11 = [make_promo(11, 0, 6, 4, 0.02), make_promo(11, 1, 8, 2, 0.06)]
    # When: the counterfactual runs
    result = stopping_counterfactual(seed_11, alpha=0.05)
    # Then: no halt point, and the final delta is still reported
    assert result["halt_promo_num"] is None
    assert result["halt_promo_idx"] is None
    assert result["peak_confirm_delta"] is None
    assert result["final_confirm_delta"] == pytest.approx(0.06)
    assert result["n_promotions_after_halt"] == 2


def test_stopping_counterfactual_rejects_an_empty_sequence() -> None:
    """A seed with no promotions is a caller error, not a silent empty result."""
    # Given / When / Then
    with pytest.raises(ValueError):
        stopping_counterfactual([], alpha=0.05)


def test_promotion_report_flags_only_the_halt_row() -> None:
    """would_halt_here marks exactly the seed's counterfactual stopping promotion."""
    # Given: seed 101
    by_seed: Dict[int, Sequence[PromotionRecord]] = {101: SEED_101}
    # When: the report is assembled
    report = promotion_report(by_seed, alpha=0.05)
    # Then: only promo #4 is the halt row, and clears/halt flags are consistent
    halted = [row for row in report["rows"] if row["would_halt_here"]]
    assert [row["promo_num"] for row in halted] == [4]
    assert report["n_promotions"] == 6
    assert report["n_seeds"] == 1
    # promo #1 (7 vs 6) is a coin toss: within the noise band and not a halt
    first = report["rows"][0]
    assert first["reversal_prob"] == 0.5
    assert first["clears_noise_band"] is False


def test_promotion_report_records_the_state_each_delta_is_measured_against() -> None:
    """A confirm delta is a marginal, so the baseline it is measured against is derived data.

    Section 4.4 says the deltas do not compound; the claim is only checkable if
    each row carries the accuracy of the state it was measured against
    (AGENTS.md §3.6 — a number in the paper resolves to a derived field).
    """
    # Given: a promotion with 7 gains and 6 losses over 100 questions, so the
    # baseline state answered 93 of them correctly
    by_seed: Dict[int, Sequence[PromotionRecord]] = {101: [make_promo(101, 0, 7, 6, 0.01)]}
    # When: the report is assembled
    row = promotion_report(by_seed, alpha=0.05)["rows"][0]
    # Then: the baseline accuracy is on the row, and the delta is not read off it
    assert row["confirm_baseline_acc"] == 0.93
    assert row["confirm_delta"] == 0.01


def test_promotion_report_is_byte_stable_across_two_runs() -> None:
    """The derived payload is a deterministic function of its inputs (AGENTS.md §3.3)."""
    # Given: a two-seed input
    by_seed: Dict[int, Sequence[PromotionRecord]] = {
        101: SEED_101,
        5: [make_promo(5, 0, 8, 0, 0.08), make_promo(5, 1, 11, 1, 0.10)],
    }
    # When: the report is serialized twice
    first = json.dumps(promotion_report(by_seed, alpha=0.05), sort_keys=True)
    second = json.dumps(promotion_report(by_seed, alpha=0.05), sort_keys=True)
    # Then: byte-identical, and the digest is stable
    assert first == second
    assert promotion_report(by_seed)["input_digest"].startswith("sha256:")
