"""Exp 4 promotion-reversal case study (ADR-017): was an accept-if-better promotion noise?

For each promoted transform in an external self-improvement loop (Nakajima's
"Regimes", arXiv:2606.10241) we hold exactly ONE held-out CONFIRM evaluation — a
paired binary vector over 100 questions, decomposed into discordant pairs
``b`` = wrong->right gains and ``c`` = right->wrong losses. There is no repeat
axis in that data (EXP4-regimes-data-check.md), so the repeat-based FIP
estimator (:mod:`mdi.stats.fip`) cannot apply; this is its parametric,
single-observation cousin — a McNemar treatment of the discordant pairs.

Two exact quantities, both from :func:`math.comb` (no scipy, no RNG, no
wall-clock — deterministic, AGENTS.md §3.3):

* :func:`mcnemar_exact` — the two-sided exact McNemar p-value: under the null
  that each discordant pair flips by a fair coin, the probability of a split at
  least this lopsided in EITHER direction.
* :func:`promotion_reversal_probability` — the one-sided exact tail
  ``P(Binom(n, 0.5) >= max(b, c))`` (which equals ``mcnemar_exact / 2`` in the
  non-degenerate case). Read as "the probability a pure-noise process would
  manufacture a net gain at least this large in the observed direction" — a
  NULL-conditioned significance tail, the papers' "per-promotion noise
  probability" (renamed 2026-08-03, ADR-017 Update: it is NOT a
  FIP, which conditions the other way, on the observed gain; under a true
  no-change null the sign-reversal chance of a fresh evaluation is 0.5 by
  symmetry regardless of this tail). The function name predates the rename
  and is kept for stability of callers and derived-JSON field names. At
  ``(7, 6)`` the tail is exactly 0.5 — chance level, the paper's headline
  over-promotion; at ``(11, 1)`` it is 0.003.

DEFINITIONAL CHOICE flagged for first-author review (no Accepted ADR fixes it;
ADR-017 §2 states BOTH framings and calls them the same "reversal probability"):
we use the one-sided McNemar tail rather than a plug-in parametric bootstrap
(model each discordant pair as Bernoulli(``b / n``), reversal = P(re-evaluated
net <= 0)). The plug-in gives 0.39 at ``(7, 6)``, which does NOT reproduce the
"coin toss" reading the paper and the data-check note both cite (0.5); the
one-sided tail does, and keeps the exact algebraic tie to the McNemar p-value.
The plug-in is also optimistically biased — it estimates the flip rate from the
very data it then reuses. First author to confirm the one-sided-tail definition
or replace it (and to fix :data:`DEFAULT_ALPHA`, the stopping band, which is an
operating point ADR-017 leaves open).

:func:`stopping_counterfactual` applies the paper's proposed plateau rule in our
language: keep the loop only while a promotion clears the noise band
(``reversal <= alpha``); the loop should have halted after the LAST such
promotion, since everything past it is within noise and only drifts the held-out
accuracy. For seed 101 that halt is promotion #4 (13 vs 4, +0.09) — the peak,
matching the author's own prediction — against a drifted final of +0.01 (7 vs 6).

Figure generation is intentionally omitted: :mod:`mdi.stats.figures` is out of
scope for ADR-017's first cut, so this module emits JSON + a table only. A
per-promotion reversal-probability bar (with the stopping point marked) is a
TODO for a later, concurrency-safe addition to the figures module.
"""

import hashlib
import json
from math import comb
from typing import Dict, List, Mapping, Optional, Sequence, TypedDict

from mdi.store import PromotionRecord

DEFAULT_ALPHA: float = 0.05
"""Noise-band significance for the stopping counterfactual. Flagged for the ADR
line: ADR-017 fixes the estimator, not this operating point."""

STOPPING_RULE: str = (
    "halt after the last promotion whose reversal_prob <= alpha; promotions past "
    "it are within the noise band (a plateau) and only drift the held-out "
    "accuracy (ADR-017 §2; Nakajima 2026 plateau rule)"
)


def _lower_tail_count(n: int, k: int) -> int:
    """Integer numerator of ``P(Binom(n, 0.5) <= k)`` over ``2**n``: ``sum_{i=0}^{k} C(n, i)``."""
    if k < 0:
        return 0
    return sum(comb(n, i) for i in range(0, min(k, n) + 1))


def promotion_reversal_probability(b: int, c: int) -> float:
    """One-sided exact McNemar tail ``P(Binom(n, 0.5) >= max(b, c))``; see module docstring.

    Equals ``P(Binom(n, 0.5) <= min(b, c))`` by the symmetry of the fair coin —
    the smaller discordant count's tail. Reads as the probability a pure-noise
    (fair-coin) process reproduces a net directional imbalance at least this
    large: the probability the accept-if-better decision is indistinguishable
    from noise / would reverse sign under a fresh evaluation. Degenerate
    ``n = 0`` (no discordant pairs, hence no evidence) returns 1.0. Monotone
    non-increasing as ``(b, c)`` separate for fixed ``n``.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    return _lower_tail_count(n, min(b, c)) / (1 << n)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs ``(b, c)``; deterministic.

    Each of the ``n = b + c`` discordant pairs is exchangeable Bernoulli(0.5)
    under the null of no directional effect; the exact two-sided binomial tail is
    ``min(1, 2 * P(Binom(n, 0.5) <= min(b, c)))``. No continuity correction, no
    scipy — an exact sum of binomial coefficients. Symmetric in its arguments;
    ``n = 0`` returns 1.0.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    two_sided = 2 * _lower_tail_count(n, min(b, c)) / (1 << n)
    return min(1.0, two_sided)


def prospective_min_majority(n_discordant: int, alpha: float = DEFAULT_ALPHA) -> Optional[int]:
    """Smallest majority count ``b`` that clears the noise band, BEFORE running the loop.

    The prospective inverse of :func:`promotion_reversal_probability`: given that a
    held-out gate will produce ``n_discordant`` discordant pairs, the promotion may
    be accepted only if the majority direction reaches ``b`` with
    ``P(Binom(n, 0.5) >= b) <= alpha``. Returns ``None`` when no majority within
    ``n_discordant`` clears the band (the gate is underpowered at this alpha —
    the loop should not promote at all, cf. Regimes seeds 11/23).

    This is the "tell them the failure bar BEFORE they run" number: for a
    100-question gate it converts directly to a minimum acceptable accuracy
    delta ``(2b - n)/100``.
    """
    if n_discordant < 0:
        raise ValueError(f"n_discordant must be non-negative, got {n_discordant}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = n_discordant
    for b in range(n // 2 + 1, n + 1):
        upper_tail = (1 << n) - _lower_tail_count(n, b - 1) if b > 0 else (1 << n)
        if upper_tail / (1 << n) <= alpha:
            return b
    return None


def prospective_min_delta(
    n_discordant: int, n_questions: int, alpha: float = DEFAULT_ALPHA
) -> Optional[float]:
    """Minimum acceptable accuracy delta for a promotion, stated in advance.

    ``(2b - n) / n_questions`` for the smallest band-clearing majority ``b``;
    ``None`` when the gate is underpowered (no acceptable promotion exists at
    this discordant count and alpha).
    """
    if n_questions <= 0:
        raise ValueError(f"n_questions must be positive, got {n_questions}")
    b = prospective_min_majority(n_discordant, alpha)
    if b is None:
        return None
    return (2 * b - n_discordant) / n_questions


class StoppingResult(TypedDict):
    """Where the accept-if-better loop should have halted for one seed (ADR-017)."""

    seed: int
    alpha: float
    n_promotions: int
    halt_promo_idx: Optional[int]  # 0-based, the last promotion clearing the band
    halt_promo_num: Optional[int]  # 1-based, for display
    halt_n_recovered: Optional[int]
    halt_n_introduced: Optional[int]
    peak_confirm_delta: Optional[float]
    final_confirm_delta: float
    n_promotions_after_halt: int
    drift_confirm_delta: Optional[float]  # final - peak (negative == drifted down)
    rule: str


class PromotionRow(TypedDict):
    """One promotion's reversal estimate plus its stopping-rule verdict."""

    seed: int
    promo_idx: int  # 0-based, as stored
    promo_num: int  # 1-based, for display
    name: str
    confirm_delta: float
    confirm_baseline_acc: float  # accuracy of the state this delta is measured against
    n_recovered: int  # b, wrong->right gains
    n_introduced: int  # c, right->wrong losses
    n_discordant: int
    mcnemar_p: float
    reversal_prob: float
    clears_noise_band: bool
    would_halt_here: bool


class PromotionReport(TypedDict):
    """Full ADR-017 payload: per-promotion rows + per-seed stopping counterfactuals."""

    alpha: float
    n_promotions: int
    n_seeds: int
    input_digest: str
    rows: List[PromotionRow]
    stopping: List[StoppingResult]


def stopping_counterfactual(
    promotions: Sequence[PromotionRecord], alpha: float = DEFAULT_ALPHA
) -> StoppingResult:
    """Where the loop should have halted for one seed's promotion sequence (ADR-017).

    Walks the promotions in order. A promotion "clears the noise band" when its
    :func:`promotion_reversal_probability` is ``<= alpha``. The counterfactual
    halt is the LAST clearing promotion — past it every promotion is
    indistinguishable from noise and only drifts the held-out accuracy.
    ``peak_confirm_delta`` is that halt promotion's ``confirm_delta``;
    ``final_confirm_delta`` is the last actual promotion's. If no promotion
    clears the band the loop should not have promoted at all (halt indices are
    ``None``).
    """
    if not promotions:
        raise ValueError("stopping_counterfactual over an empty promotion sequence")
    seed = promotions[0]["seed"]
    final_delta = float(promotions[-1]["confirm_delta"])
    halt_idx: Optional[int] = None
    for idx, promo in enumerate(promotions):
        if promotion_reversal_probability(promo["n_recovered"], promo["n_introduced"]) <= alpha:
            halt_idx = idx
    if halt_idx is None:
        return StoppingResult(
            seed=seed,
            alpha=alpha,
            n_promotions=len(promotions),
            halt_promo_idx=None,
            halt_promo_num=None,
            halt_n_recovered=None,
            halt_n_introduced=None,
            peak_confirm_delta=None,
            final_confirm_delta=final_delta,
            n_promotions_after_halt=len(promotions),
            drift_confirm_delta=None,
            rule=STOPPING_RULE,
        )
    halt = promotions[halt_idx]
    peak_delta = float(halt["confirm_delta"])
    return StoppingResult(
        seed=seed,
        alpha=alpha,
        n_promotions=len(promotions),
        halt_promo_idx=halt_idx,
        halt_promo_num=halt_idx + 1,
        halt_n_recovered=halt["n_recovered"],
        halt_n_introduced=halt["n_introduced"],
        peak_confirm_delta=peak_delta,
        final_confirm_delta=final_delta,
        n_promotions_after_halt=len(promotions) - halt_idx - 1,
        drift_confirm_delta=final_delta - peak_delta,
        rule=STOPPING_RULE,
    )


def _digest(by_seed: Mapping[int, Sequence[PromotionRecord]]) -> str:
    """SHA-256 over the canonical ``(seed, promo_idx, b, c, delta, name)`` projection."""
    projection = [
        [
            promo["seed"],
            promo["promo_idx"],
            promo["n_recovered"],
            promo["n_introduced"],
            round(float(promo["confirm_delta"]), 12),
            promo["name"],
        ]
        for seed in sorted(by_seed)
        for promo in by_seed[seed]
    ]
    payload = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def promotion_report(
    by_seed: Mapping[int, Sequence[PromotionRecord]], alpha: float = DEFAULT_ALPHA
) -> PromotionReport:
    """Assemble per-promotion rows and per-seed stopping counterfactuals (pure, deterministic).

    Rows are emitted seed-ascending then in promotion order; each carries its
    exact McNemar p, one-sided reversal probability, whether it clears the noise
    band, and whether it is its seed's counterfactual halt point.
    """
    stopping = [stopping_counterfactual(by_seed[seed], alpha) for seed in sorted(by_seed)]
    halt_lookup: Dict[int, Optional[int]] = {s["seed"]: s["halt_promo_idx"] for s in stopping}
    rows: List[PromotionRow] = []
    for seed in sorted(by_seed):
        for promo in by_seed[seed]:
            b, c = promo["n_recovered"], promo["n_introduced"]
            reversal = promotion_reversal_probability(b, c)
            rows.append(
                PromotionRow(
                    seed=seed,
                    promo_idx=promo["promo_idx"],
                    promo_num=promo["promo_idx"] + 1,
                    name=promo["name"],
                    confirm_delta=float(promo["confirm_delta"]),
                    confirm_baseline_acc=(
                        sum(1 for ok in promo["base_correct"] if ok) / len(promo["base_correct"])
                    ),
                    n_recovered=b,
                    n_introduced=c,
                    n_discordant=b + c,
                    mcnemar_p=mcnemar_exact(b, c),
                    reversal_prob=reversal,
                    clears_noise_band=reversal <= alpha,
                    would_halt_here=(halt_lookup.get(seed) == promo["promo_idx"]),
                )
            )
    return PromotionReport(
        alpha=alpha,
        n_promotions=len(rows),
        n_seeds=len(by_seed),
        input_digest=_digest(by_seed),
        rows=rows,
        stopping=stopping,
    )
