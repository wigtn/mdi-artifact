"""Tests for the shared seeded bootstrap (mdi.stats.bootstrap)."""

from typing import List, Optional, Sequence

import pytest

from mdi.stats.bootstrap import (
    DEFAULT_CI_LEVEL,
    bootstrap_scalar,
    bootstrap_vector,
    make_rng,
    percentile,
    resample_indices,
)


def _mean(values: Sequence[float]) -> Optional[float]:
    """Sample mean, or ``None`` for an empty sample."""
    return sum(values) / len(values) if values else None


def test_bootstrap_result_is_reproducible_under_a_fixed_seed() -> None:
    """Two bootstraps with the same seed must agree bit for bit."""
    # Given: a fixed sample and a fixed seed
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    # When: the same bootstrap runs twice
    first = bootstrap_scalar(values, _mean, seed=42, n_resamples=500)
    second = bootstrap_scalar(values, _mean, seed=42, n_resamples=500)
    # Then: point estimate, interval and metadata are identical
    assert first == second
    assert first["meta"] == {
        "seed": 42,
        "n_resamples": 500,
        "ci_level": DEFAULT_CI_LEVEL,
        "method": "percentile",
        "n_clusters": len(values),
        "degenerate": False,
    }


def test_bootstrap_interval_is_undefined_when_a_single_cluster() -> None:
    """One cluster reproduces itself every replicate — report no interval, not a zero-width one."""
    # Given: a vector statistic over exactly one cluster
    clusters = [[1.0, 2.0, 3.0]]

    def first_mean(sample: Sequence[List[float]]) -> Sequence[Optional[float]]:
        pooled = [value for cluster in sample for value in cluster]
        return [_mean(pooled)]

    # When: the cluster bootstrap runs
    result = bootstrap_vector(clusters, first_mean, seed=42, n_resamples=100)[0]
    # Then: the point estimate stands but the interval is undefined and flagged
    assert result["point"] == pytest.approx(2.0)
    assert result["ci_lo"] is None
    assert result["ci_hi"] is None
    assert result["meta"]["degenerate"] is True
    assert result["meta"]["n_clusters"] == 1


def test_bootstrap_interval_is_defined_when_clusters_suffice() -> None:
    """Two or more clusters must still produce a real interval."""
    # Given: the same statistic over several clusters
    clusters = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

    def pooled_mean(sample: Sequence[List[float]]) -> Sequence[Optional[float]]:
        pooled = [value for cluster in sample for value in cluster]
        return [_mean(pooled)]

    # When: the cluster bootstrap runs
    result = bootstrap_vector(clusters, pooled_mean, seed=42, n_resamples=200)[0]
    # Then: the interval is a genuine band
    assert result["meta"]["degenerate"] is False
    assert result["ci_lo"] is not None and result["ci_hi"] is not None
    assert result["ci_hi"] > result["ci_lo"]


def test_bootstrap_result_differs_under_a_different_seed() -> None:
    """A different seed must actually redraw (the seed is not ignored)."""
    # Given: a fixed sample
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    # When: two bootstraps use different seeds
    first = bootstrap_scalar(values, _mean, seed=1, n_resamples=200)
    second = bootstrap_scalar(values, _mean, seed=2, n_resamples=200)
    # Then: the point estimate is unchanged but the interval moves
    assert first["point"] == second["point"]
    assert (first["ci_lo"], first["ci_hi"]) != (second["ci_lo"], second["ci_hi"])


def test_bootstrap_interval_brackets_the_point_estimate_for_a_mean() -> None:
    """The percentile interval must contain the observed statistic."""
    # Given: a sample with known mean 4.5
    values = [float(index) for index in range(1, 9)]
    # When: bootstrapping the mean
    result = bootstrap_scalar(values, _mean, seed=7, n_resamples=400)
    # Then: the interval brackets the point estimate
    assert result["point"] == pytest.approx(4.5)
    assert result["ci_lo"] is not None and result["ci_hi"] is not None
    assert result["ci_lo"] <= 4.5 <= result["ci_hi"]
    assert result["n_effective"] == 400


def test_bootstrap_vector_skips_undefined_entries_only_for_that_entry() -> None:
    """A ``None`` entry drops that dimension's replicate, not the whole resample."""

    # Given: a statistic whose second entry is undefined
    def statistic(sample: Sequence[float]) -> List[Optional[float]]:
        return [_mean(sample), None]

    # When: the cluster bootstrap runs
    results = bootstrap_vector([1.0, 2.0, 3.0], statistic, seed=3, n_resamples=50)
    # Then: entry 0 has replicates and entry 1 has none
    assert results[0]["n_effective"] == 50
    assert results[1]["n_effective"] == 0
    assert results[1]["ci_lo"] is None and results[1]["ci_hi"] is None


def test_bootstrap_raises_on_an_empty_cluster_list() -> None:
    """An empty sample is a caller error, not a silent ``None``."""
    # Given/When/Then: bootstrapping nothing raises
    with pytest.raises(ValueError, match="empty cluster list"):
        bootstrap_scalar([], _mean, seed=1, n_resamples=10)


def test_percentile_interpolates_between_order_statistics() -> None:
    """Percentiles use the type-7 (NumPy default) linear interpolation."""
    # Given: four ordered values
    values = [1.0, 2.0, 3.0, 4.0]
    # When/Then: quartiles interpolate between order statistics
    assert percentile(values, 0.0) == pytest.approx(1.0)
    assert percentile(values, 0.5) == pytest.approx(2.5)
    assert percentile(values, 0.25) == pytest.approx(1.75)
    assert percentile(values, 1.0) == pytest.approx(4.0)


def test_percentile_raises_when_q_is_out_of_range() -> None:
    """A quantile outside [0, 1] is rejected."""
    # Given/When/Then
    with pytest.raises(ValueError, match=r"q must be in \[0, 1\]"):
        percentile([1.0, 2.0], 1.5)


def test_resample_indices_are_stable_for_a_given_seed() -> None:
    """Index draws depend only on the seed and the call order."""
    # Given: two identically seeded RNGs
    first = resample_indices(6, make_rng(11))
    second = resample_indices(6, make_rng(11))
    # When/Then: the drawn index sequences match and stay in range
    assert first == second
    assert len(first) == 6
    assert all(0 <= index < 6 for index in first)
