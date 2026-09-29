"""Judge-response parsing layer: raw_response -> parsed_score (FR-003 derived layer).

Parse-failure policy (ADR-011 D7 via PRD §5.2):

- **No re-prompting.** Changing the prompt breaks the fixed-env repetition
  premise.
- Failed parses are preserved as records with ``parse_ok: false`` — never
  deleted or edited (raw store is immutable, AGENTS.md §3.1).
- Valid repeats are recovered by redrawing with a *new* ``repeat_idx`` in the
  same env, up to an attempt cap of 2N (:func:`max_attempts`).
- The per-env parse failure rate is itself an analysis output
  (:func:`parse_failure_rate`) — it is part of judge noise.

Parsed scores are a regenerable derived layer: fixing a parser bug means
re-running the parser over raw records, never editing stored records.
"""

import json
import re
from typing import Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple

# Scale -> (minimum, maximum) admissible score. Values outside the range are a
# parse failure, not a clamp: a judge that answers "7" on a 1-5 rubric did not
# produce a Likert-5 score.
SCALE_RANGES: Dict[str, Tuple[float, float]] = {
    "likert5": (1.0, 5.0),
    "likert10": (1.0, 10.0),
    "score100": (0.0, 100.0),
}

DEFAULT_MAX_ATTEMPTS_FACTOR: int = 2

_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_LABELLED = re.compile(
    rf"\b(?:score|rating|rate|grade|점수)\b\s*(?:is|=|:)?\s*({_NUMBER})",
    re.IGNORECASE,
)
_FRACTION = re.compile(rf"({_NUMBER})\s*(?:/|\bout\s+of\b)\s*({_NUMBER})", re.IGNORECASE)
_ANY_NUMBER = re.compile(_NUMBER)
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class ParseResult(NamedTuple):
    """Outcome of parsing one judge response."""

    ok: bool
    score: Optional[float]
    reason: str


def max_attempts(n_repeats: int, factor: int = DEFAULT_MAX_ATTEMPTS_FACTOR) -> int:
    """Attempt cap for one (env, item, system) cell — ADR-011 D7's 2N rule."""
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    if factor < 1:
        raise ValueError("max_attempts_factor must be >= 1")
    return n_repeats * factor


def _in_range(value: float, scale: str) -> bool:
    """True when *value* lies inside the admissible range for *scale*."""
    low, high = SCALE_RANGES[scale]
    return low <= value <= high


def _from_json(text: str, scale: str) -> Optional[float]:
    """Extract a score from a JSON object embedded in *text* (``{"score": 4}``)."""
    match = _JSON_BLOCK.search(text)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    for key in ("score", "rating", "value"):
        raw = obj.get(key)
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            value = float(raw)
            if _in_range(value, scale):
                return value
        if isinstance(raw, str):
            candidate = _ANY_NUMBER.search(raw)
            if candidate is not None:
                value = float(candidate.group(0))
                if _in_range(value, scale):
                    return value
    return None


def _from_fraction(text: str, scale: str) -> Optional[float]:
    """Extract ``4/5`` / ``4 out of 5`` forms whose denominator matches the scale maximum."""
    high = SCALE_RANGES[scale][1]
    for match in _FRACTION.finditer(text):
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator == high and _in_range(numerator, scale):
            return numerator
    return None


def _from_label(text: str, scale: str) -> Optional[float]:
    """Extract a labelled score (``Score: 4``)."""
    for match in _LABELLED.finditer(text):
        value = float(match.group(1))
        if _in_range(value, scale):
            return value
    return None


def _from_sole_number(text: str, scale: str) -> Optional[float]:
    """Extract the number when the response contains exactly one numeric token."""
    numbers = _ANY_NUMBER.findall(text)
    if len(numbers) != 1:
        return None
    value = float(numbers[0])
    return value if _in_range(value, scale) else None


def parse_response(raw_response: str, scale: str) -> ParseResult:
    """Parse one verbatim judge response for *scale*, reporting why it failed.

    Strategies are tried in decreasing order of explicitness: embedded JSON, a
    labelled score, an ``x/max`` fraction, then a lone number. Anything else —
    including an in-text number that is ambiguous, or a value outside the
    scale's range — is a parse failure (``parse_ok: false``, never re-prompted).
    """
    if scale not in SCALE_RANGES:
        raise ValueError(f"unknown scale: {scale!r} (known: {sorted(SCALE_RANGES)})")
    text = raw_response.strip()
    if not text:
        return ParseResult(ok=False, score=None, reason="empty_response")
    for extract in (_from_json, _from_label, _from_fraction, _from_sole_number):
        value = extract(text, scale)
        if value is not None:
            return ParseResult(ok=True, score=value, reason="ok")
    return ParseResult(ok=False, score=None, reason="no_score_in_range")


def parse_score(raw_response: str, scale: str) -> Optional[float]:
    """Parse one verbatim judge response into a numeric score for *scale*.

    Returns ``None`` on parse failure (recorded as ``parse_ok: false``).
    """
    return parse_response(raw_response, scale).score


class ParseFailureStat(NamedTuple):
    """Per-environment parse-failure statistic (an Exp 1 analysis output, ADR-011 D7)."""

    env_id: str
    attempts: int
    failures: int

    @property
    def rate(self) -> float:
        """Failure rate over attempts; 0.0 when there were no attempts."""
        return self.failures / self.attempts if self.attempts else 0.0


def parse_failure_rate(records: Iterable[Mapping[str, object]]) -> List[ParseFailureStat]:
    """Per-env parse-failure rate over raw score records, sorted by ``env_id``.

    Accepts any mapping carrying ``env_id`` and ``parse_ok`` (i.e. every
    :class:`~mdi.store.ScoreRecord`). The failure rate is a finding in its own
    right, not an error condition.
    """
    attempts: Dict[str, int] = {}
    failures: Dict[str, int] = {}
    for record in records:
        env_id = str(record["env_id"])
        attempts[env_id] = attempts.get(env_id, 0) + 1
        if not bool(record["parse_ok"]):
            failures[env_id] = failures.get(env_id, 0) + 1
    return [
        ParseFailureStat(env_id=env_id, attempts=attempts[env_id], failures=failures.get(env_id, 0))
        for env_id in sorted(attempts)
    ]
