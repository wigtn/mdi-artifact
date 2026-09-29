"""Inter-coder agreement (Cohen's kappa) — stub (FR-010: Phase 1).

Kappa is computed per core field with a 0.85 gate each; below-gate fields
produce a disagreement list for manual revision (one re-coding round max, then
the measured kappa is reported as-is). Agents compute kappa over human-produced
sheets only — never produce or edit the codings themselves (AGENTS.md §3.5).
"""

from typing import Any, Dict, List


def compute_kappa(
    coder_a_rows: List[Dict[str, Any]], coder_b_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute per-field Cohen's kappa and the disagreement list. Stub."""
    raise NotImplementedError("FR-010: Phase 1")
