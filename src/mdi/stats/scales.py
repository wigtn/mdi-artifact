"""ADR-001 scale-normalized percentage-point (%p) notation helpers.

ADR-001 Update (2026-08-01, first-author decision): the paper's primary unit is
the scale-normalized percentage point — a raw score difference divided by the
width of its scale, times 100 (likert5's 0.294 -> 7.35 %p). Raw-scale values
are preserved in derived JSON for reproducibility and audit; figures and tables
render %p as the primary unit.

The scale width is derived from the record's ``scale`` field: likert5 spans
1..5 (width 4), likert10 spans 1..10 (width 9), score100 spans 0..100
(width 100). An unknown scale raises — silently guessing a width would corrupt
every rendered number.
"""

from typing import Dict, Optional

SCALE_WIDTHS: Dict[str, float] = {
    "likert5": 4.0,
    "likert10": 9.0,
    "score100": 100.0,
}


def scale_width(scale: str) -> float:
    """Width of *scale* in raw score units; raise ``ValueError`` on an unknown scale."""
    try:
        return SCALE_WIDTHS[scale]
    except KeyError:
        raise ValueError(
            f"unknown scale {scale!r}: no ADR-001 width is defined (known: {sorted(SCALE_WIDTHS)})"
        ) from None


def to_pp(value: float, scale: str) -> float:
    """Convert a raw score difference to scale-normalized percentage points (ADR-001)."""
    return 100.0 * value / scale_width(scale)


def maybe_pp(value: Optional[float], scale: str) -> Optional[float]:
    """:func:`to_pp` that passes ``None`` through (undefined stays undefined)."""
    return None if value is None else to_pp(value, scale)


def from_pp(value_pp: float, scale: str) -> float:
    """Convert scale-normalized percentage points back to a raw score difference."""
    return value_pp * scale_width(scale) / 100.0
