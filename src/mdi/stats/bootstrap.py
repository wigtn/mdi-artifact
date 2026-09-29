"""Shared seeded empirical bootstrap for the analysis layer (FR-006/007/009/011/013).

Every bootstrap in this project goes through here so that resampling is (a)
seeded from an explicit integer, (b) drawn in a fixed traversal order, and (c)
self-describing: each result carries the seed, the number of resamples, the CI
level and the method in its metadata (AGENTS.md §3.3, PRD §4.1).

Determinism note: draws use :class:`random.Random` (Mersenne Twister), whose
stream is reproducible across runs, processes and platforms — deliberately not
a NumPy ``Generator``, whose bit stream carries no equivalent cross-version
guarantee.

Methodology defaults flagged for first-author review (no Accepted ADR covers
them yet): :data:`DEFAULT_N_RESAMPLES`, :data:`DEFAULT_CI_LEVEL`,
:data:`DEFAULT_SEED`, the percentile (rather than BCa) interval, and — for
clustered data — resampling whole clusters.
"""

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict, TypeVar

DEFAULT_N_RESAMPLES: int = 2000
DEFAULT_CI_LEVEL: float = 0.95
DEFAULT_SEED: int = 20260731
BOOTSTRAP_METHOD: str = "percentile"

#: A cluster bootstrap resamples whole clusters, so a single cluster reproduces
#: itself in every replicate and the interval collapses onto the point estimate.
#: A zero-width interval is not a confidence interval — it is the absence of one
#: wearing its costume, and reporting it would violate the project's own
#: protocol (AGENTS.md §3.6). Below this many clusters the interval is reported
#: as undefined (``None``) and ``meta["degenerate"]`` is set.
#:
#: Substituting a different cluster unit (items rather than system pairs, say)
#: when pairs are scarce would change what the interval means, so it is a
#: methodology choice for the ADR line — not something this module decides.
MIN_CLUSTERS_FOR_CI: int = 2

T = TypeVar("T")


class BootstrapMeta(TypedDict, total=False):
    """Self-describing provenance for one bootstrap computation."""

    seed: int
    n_resamples: int
    ci_level: float
    method: str
    n_clusters: int
    degenerate: bool


class BootstrapResult(TypedDict):
    """Point estimate plus a percentile confidence interval."""

    point: Optional[float]
    ci_lo: Optional[float]
    ci_hi: Optional[float]
    n_effective: int
    meta: BootstrapMeta


def make_rng(seed: int) -> random.Random:
    """Return the project's seeded RNG (Mersenne Twister)."""
    return random.Random(seed)


def bootstrap_meta(
    seed: int,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> BootstrapMeta:
    """Build the metadata block recorded next to every bootstrap output."""
    return BootstrapMeta(
        seed=seed,
        n_resamples=n_resamples,
        ci_level=ci_level,
        method=BOOTSTRAP_METHOD,
    )


def resample_indices(n: int, rng: random.Random) -> List[int]:
    """Draw *n* indices in ``[0, n)`` with replacement (one empirical resample)."""
    return [rng.randrange(n) for _ in range(n)]


def percentile(values: Sequence[float], q: float, *, assume_sorted: bool = False) -> float:
    """Linear-interpolation percentile of *values* for ``q`` in ``[0, 1]``.

    Order-statistic interpolation (the "type 7" definition, NumPy's default), so
    the interval is a deterministic function of the resample set. Callers that
    take several quantiles of one large sample can pre-sort it once and pass
    ``assume_sorted=True``.
    """
    if not values:
        raise ValueError("percentile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    ordered = list(values) if assume_sorted else sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _interval(
    replicates: Sequence[float], ci_level: float
) -> Tuple[Optional[float], Optional[float]]:
    """Percentile interval at *ci_level* over the bootstrap replicates."""
    if not replicates:
        return None, None
    tail = (1.0 - ci_level) / 2.0
    return percentile(replicates, tail), percentile(replicates, 1.0 - tail)


def bootstrap_vector(
    clusters: Sequence[T],
    statistic: Callable[[Sequence[T]], Sequence[Optional[float]]],
    *,
    seed: int,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> List[BootstrapResult]:
    """Cluster bootstrap for a vector-valued *statistic* (e.g. one FIP per delta bin).

    Whole clusters — pairs, items, or whatever unit carries the dependence — are
    resampled with replacement; *statistic* maps a resampled cluster list to a
    fixed-length vector whose entries may be ``None`` (undefined for that
    resample, e.g. an empty bin), in which case that replicate is skipped for
    that entry only.

    With fewer than :data:`MIN_CLUSTERS_FOR_CI` clusters every replicate
    reproduces the input, so the interval is reported as undefined rather than
    as a zero-width band; ``meta["degenerate"]`` marks those results.
    """
    if not clusters:
        raise ValueError("bootstrap over an empty cluster list")
    point = list(statistic(clusters))
    degenerate = len(clusters) < MIN_CLUSTERS_FOR_CI
    replicates: List[List[float]] = [[] for _ in point]
    rng = make_rng(seed)
    if not degenerate:
        for _ in range(n_resamples):
            sample = [clusters[i] for i in resample_indices(len(clusters), rng)]
            values = statistic(sample)
            if len(values) != len(point):
                raise ValueError("statistic returned vectors of varying length")
            for index, value in enumerate(values):
                if value is not None:
                    replicates[index].append(value)
    meta = bootstrap_meta(seed=seed, n_resamples=n_resamples, ci_level=ci_level)
    meta["n_clusters"] = len(clusters)
    meta["degenerate"] = degenerate
    results: List[BootstrapResult] = []
    for index, value in enumerate(point):
        ci_lo, ci_hi = (None, None) if degenerate else _interval(replicates[index], ci_level)
        results.append(
            BootstrapResult(
                point=value,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                n_effective=len(replicates[index]),
                meta=meta,
            )
        )
    return results


def bootstrap_scalar(
    values: Sequence[T],
    statistic: Callable[[Sequence[T]], Optional[float]],
    *,
    seed: int,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> BootstrapResult:
    """Empirical bootstrap for a scalar *statistic* over exchangeable observations."""

    def vector(sample: Sequence[T]) -> Sequence[Optional[float]]:
        return [statistic(sample)]

    return bootstrap_vector(
        values,
        vector,
        seed=seed,
        n_resamples=n_resamples,
        ci_level=ci_level,
    )[0]


def bootstrap_vanish_rate(seed: int) -> Dict[str, Any]:
    """Compute the bootstrap vanish rate over the merged census sheet. Stub.

    FR-011 (Exp 4). Left as a Phase-1 stub here: ADR-013 (first-author decision
    Option A) replaces the census with self-improvement-loop re-measurement, so
    the Exp 4 estimator is blocked on that ADR rather than on this module.
    """
    raise NotImplementedError("FR-011: Phase 1 (scope pending ADR-013)")
