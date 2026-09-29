"""Guard tests for the score record schema v2 (PRD §5.2, ADR-009/006)."""

from mdi.store import SCHEMA_VERSION, ScoreRecord, Usage

# Exact field-name set from PRD §5.2 schema v2 — do not edit without a new ADR.
EXPECTED_SCORE_RECORD_FIELDS = {
    "schema_version",
    "run_id",
    "env_id",
    "judge_model",
    "judge_model_version",
    "judge_tier",
    "serving_engine",
    "quantization",
    "seed",
    "prompt_id",
    "paraphrase_id",
    "temperature",
    "scale",
    "task",
    "benchmark",
    "item_id",
    "system_id",
    "repeat_idx",
    "request_payload_hash",
    "raw_response",
    "parsed_score",
    "parse_ok",
    "usage",
    "cost_usd",
    "ts",
}


def test_score_record_fields_match_prd_schema_v2() -> None:
    """ScoreRecord field names must match PRD §5.2 schema v2 exactly."""
    # Given: the PRD §5.2 schema v2 field-name set
    # When: compared against the ScoreRecord TypedDict annotations
    # Then: the sets are identical (no missing, renamed, or extra fields)
    assert set(ScoreRecord.__annotations__) == EXPECTED_SCORE_RECORD_FIELDS


def test_schema_version_constant_is_2() -> None:
    """SCHEMA_VERSION must be pinned to 2 (ADR-009 revision)."""
    # Given / When / Then
    assert SCHEMA_VERSION == 2


def test_usage_fields_match_prd_schema_v2() -> None:
    """Usage sub-record must carry exactly in_tokens and out_tokens."""
    # Given / When / Then
    assert set(Usage.__annotations__) == {"in_tokens", "out_tokens"}
