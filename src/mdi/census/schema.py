"""Census coding-sheet schema (FR-010, PRD §5.2).

One coding-sheet row per audited paper. The field list is fixed by PRD §5.2
(Card et al. meta-analysis template, ADR-011 D8). Unreported fields are coded
per the Strict Rule: not explicitly reported -> "unreported"
(see :mod:`mdi.census.strict_rule`).
"""

from typing import Any, Dict, Tuple

CENSUS_CODING_FIELDS: Tuple[str, ...] = (
    "paper_id",
    "venue",
    "year",
    "claims_improvement",
    "delta_value",
    "delta_metric",
    "judge_model",
    "n_runs_reported",
    "ci_reported",
    "seed_reported",
    "scale_type",
    "coder_id",
    "notes",
)


def validate_coding_row(row: Dict[str, Any]) -> None:
    """Validate one coding-sheet row against the schema. Stub."""
    raise NotImplementedError("FR-010: Phase 1")
