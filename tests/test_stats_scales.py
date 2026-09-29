"""Tests for the ADR-001 scale-normalized percentage-point (%p) helpers."""

import pytest

from mdi.stats.scales import maybe_pp, scale_width, to_pp


def test_pp_conversion_is_correct_per_scale() -> None:
    """Widths are derived from the scale field: likert5 -> 4, likert10 -> 9, score100 -> 100."""
    # Given: the ADR-001 update's own example (likert5's 0.294 -> 7.35 %p)
    # When/Then: each scale divides by its width and multiplies by 100
    assert to_pp(0.294, "likert5") == pytest.approx(7.35)
    assert to_pp(4.0, "likert5") == pytest.approx(100.0)
    assert to_pp(0.9, "likert10") == pytest.approx(10.0)
    assert to_pp(9.0, "likert10") == pytest.approx(100.0)
    assert to_pp(25.0, "score100") == pytest.approx(25.0)
    assert scale_width("likert5") == 4.0
    assert scale_width("likert10") == 9.0
    assert scale_width("score100") == 100.0


def test_unknown_scale_raises_instead_of_guessing_a_width() -> None:
    """A scale without an ADR-001 width must fail loudly, never render a wrong number."""
    # Given: a scale the ADR does not cover
    # When/Then: conversion refuses
    with pytest.raises(ValueError, match="unknown scale 'elo'"):
        scale_width("elo")
    with pytest.raises(ValueError, match="unknown scale"):
        to_pp(1.0, "pairwise")


def test_maybe_pp_passes_none_through() -> None:
    """An undefined raw value stays undefined in %p (no fake zero)."""
    # Given/When/Then
    assert maybe_pp(None, "likert5") is None
    assert maybe_pp(2.0, "likert5") == pytest.approx(50.0)
