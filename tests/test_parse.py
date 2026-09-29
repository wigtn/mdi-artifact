"""Tests for the scale-specific response parsers and the parse-failure statistic."""

from typing import Any, Dict, List, Optional

import pytest

from mdi.parse import (
    ParseFailureStat,
    max_attempts,
    parse_failure_rate,
    parse_response,
    parse_score,
)

LIKERT5_OK = [
    ('{"score": 4}', 4.0),
    ('Here is my answer:\n{"score": 3, "rationale": "ok"}', 3.0),
    ("Score: 5", 5.0),
    ("score = 2", 2.0),
    ("Rating: 4.5", 4.5),
    ("4/5", 4.0),
    ("I would say 4 out of 5", 4.0),
    ("2", 2.0),
    ("  3  ", 3.0),
]

LIKERT5_FAIL = [
    "",
    "I cannot rate this summary.",
    "Score: 9",
    "0",
    "The article mentions 12 people and 3 cars.",
]


@pytest.mark.parametrize(("raw", "expected"), LIKERT5_OK, ids=[r for r, _ in LIKERT5_OK])
def test_parse_score_returns_value_when_response_carries_a_likert5_score(
    raw: str, expected: float
) -> None:
    """A likert5 response in any supported shape parses to its numeric score."""
    # Given: a judge response carrying a 1-5 score
    # When: it is parsed on the likert5 scale
    result = parse_response(raw, "likert5")
    # Then: the score is recovered and the parse is marked ok
    assert result.ok is True
    assert result.score == expected


@pytest.mark.parametrize("raw", LIKERT5_FAIL, ids=[repr(r) for r in LIKERT5_FAIL])
def test_parse_score_returns_none_when_response_has_no_in_range_score(raw: str) -> None:
    """Empty, refusing, out-of-range and ambiguous responses are parse failures."""
    # Given: a judge response with no unambiguous in-range score
    # When: it is parsed on the likert5 scale
    result = parse_response(raw, "likert5")
    # Then: the parse fails (recorded as parse_ok false, never re-prompted)
    assert result.ok is False
    assert result.score is None
    assert parse_score(raw, "likert5") is None


def test_parse_score_accepts_full_range_when_scale_is_score100() -> None:
    """score100 admits values a likert5 parse would reject."""
    # Given: a response of 87
    raw = "Score: 87"
    # When: parsed on each scale
    # Then: score100 accepts it, likert5 does not
    assert parse_score(raw, "score100") == 87.0
    assert parse_score(raw, "likert5") is None


def test_parse_response_raises_when_scale_is_unknown() -> None:
    """An undeclared scale is a config error, not a silent parse failure."""
    # Given / When / Then
    with pytest.raises(ValueError, match="unknown scale"):
        parse_response('{"score": 3}', "likert7")


def test_max_attempts_is_two_n_when_factor_is_default() -> None:
    """The redraw cap is 2N (ADR-011 D7)."""
    # Given: a target of 10 valid repeats
    # When: the attempt cap is computed with the default factor
    # Then: it is 2N
    assert max_attempts(10) == 20
    assert max_attempts(8) == 16


def test_parse_failure_rate_reports_per_env_rate_when_records_mix_outcomes() -> None:
    """The per-env parse-failure rate is computable from raw records (ADR-011 D7)."""
    # Given: four records in env A (one failure) and two in env B (none)
    records: List[Dict[str, Any]] = [
        {"env_id": "e_a", "parse_ok": True},
        {"env_id": "e_a", "parse_ok": True},
        {"env_id": "e_a", "parse_ok": False},
        {"env_id": "e_a", "parse_ok": True},
        {"env_id": "e_b", "parse_ok": True},
        {"env_id": "e_b", "parse_ok": True},
    ]
    # When: the failure rate is computed
    stats = parse_failure_rate(records)
    # Then: each env reports its own rate, sorted by env_id
    assert stats == [
        ParseFailureStat(env_id="e_a", attempts=4, failures=1),
        ParseFailureStat(env_id="e_b", attempts=2, failures=0),
    ]
    assert stats[0].rate == 0.25
    assert stats[1].rate == 0.0


def test_parse_failure_rate_is_zero_when_there_are_no_records() -> None:
    """No records means no rate, not a division by zero."""
    # Given / When
    empty: List[Dict[str, Any]] = []
    stats = parse_failure_rate(empty)
    # Then
    assert stats == []
    assert ParseFailureStat(env_id="e_x", attempts=0, failures=0).rate == 0.0


def test_parse_score_prefers_json_when_response_also_contains_prose_numbers() -> None:
    """Explicit JSON wins over incidental numbers in the rationale."""
    # Given: a response whose prose mentions other numbers
    raw = 'The summary drops 3 of 5 key facts.\n{"score": 2}'
    # When / Then: the JSON score is used
    value: Optional[float] = parse_score(raw, "likert5")
    assert value == 2.0
