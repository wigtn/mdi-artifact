"""Statistical analyses over the raw score store (FR-005..FR-009, FR-011, FR-013).

All analyses read the raw/derived stores only, are seeded and sorted for
byte-identical reproducibility (AGENTS.md §3.3), and write derived artifacts —
never back into ``data/raw/``.
"""
