"""Strict Rule merge of independent codings — stub (FR-010: Phase 1).

On disagreement, the Strict Rule applies: if a value is not explicitly reported
in the paper, it is coded as "unreported". Merging happens only after the kappa
gate (see :mod:`mdi.census.kappa`).
"""

from typing import Any, Dict, List


def merge_strict(
    coder_a_rows: List[Dict[str, Any]], coder_b_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge two independent coding sheets under the Strict Rule. Stub."""
    raise NotImplementedError("FR-010: Phase 1")
