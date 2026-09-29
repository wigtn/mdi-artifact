"""Record grouping helpers shared by the analysis layer (FR-005..FR-009).

Pure functions over :class:`mdi.store.ScoreRecord`: no network, no API calls,
no wall-clock, no environment lookups. Every grouping is sorted and every
returned container is ordered deterministically, so downstream analyses and
``mdi report all`` are byte-stable (AGENTS.md §3.3).

Only ``parse_ok`` records carrying a non-null ``parsed_score`` enter an
analysis; failed parses are preserved and reported as a per-env failure rate,
which is itself an analysis output (ADR-011 D7).
"""

import hashlib
import json
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, TypedDict

from mdi.store import SCHEMA_VERSION, ScoreRecord


class SchemaMismatchError(ValueError):
    """A record's ``schema_version`` differs from :data:`mdi.store.SCHEMA_VERSION`."""


class DuplicateRepeatError(ValueError):
    """Two scored records share one call key with conflicting scores.

    The call key ``(env_id, item_id, system_id, repeat_idx)`` is idempotent by
    construction (FR-002), so a conflict is a store integrity problem.
    """


class CellKey(NamedTuple):
    """Identity of one repeated-scoring cell: a fixed ``(env_id, item_id, system_id)``."""

    env_id: str
    item_id: str
    system_id: str


class EnvMeta(TypedDict):
    """Environment-level metadata carried by every record of that env."""

    env_id: str
    task: str
    scale: str
    benchmark: str
    judge_model: str
    judge_model_version: str
    temperature: float
    n_records: int
    n_scored: int
    parse_failure_rate: float


def _record_sort_key(record: ScoreRecord) -> Tuple[str, str, str, int, str]:
    """Total order over records: env, item, system, repeat, run_id."""
    return (
        record["env_id"],
        record["item_id"],
        record["system_id"],
        record["repeat_idx"],
        record["run_id"],
    )


def check_schema(records: Iterable[ScoreRecord]) -> None:
    """Raise :class:`SchemaMismatchError` if any record is not schema v\\ :data:`SCHEMA_VERSION`.

    Analyses must never silently mix schema versions — a migration is a separate,
    explicit step (PRD §4.1).
    """
    for record in records:
        version = record["schema_version"]
        if version != SCHEMA_VERSION:
            raise SchemaMismatchError(
                f"record schema_version={version} but this build expects {SCHEMA_VERSION} "
                f"(run_id={record['run_id']}, env_id={record['env_id']})"
            )


def sorted_records(records: Iterable[ScoreRecord]) -> List[ScoreRecord]:
    """Return the records in the canonical analysis order (deterministic, stable)."""
    return sorted(records, key=_record_sort_key)


def scored_records(records: Iterable[ScoreRecord]) -> List[ScoreRecord]:
    """Return only records with a usable score (``parse_ok`` and non-null ``parsed_score``)."""
    return [r for r in sorted_records(records) if r["parse_ok"] and r["parsed_score"] is not None]


def filter_records(
    records: Iterable[ScoreRecord],
    *,
    env_id: Optional[str] = None,
    task: Optional[str] = None,
    scale: Optional[str] = None,
    system_ids: Optional[Sequence[str]] = None,
) -> List[ScoreRecord]:
    """Filter records by env / task / scale / systems, preserving canonical order."""
    systems = set(system_ids) if system_ids is not None else None
    out: List[ScoreRecord] = []
    for record in sorted_records(records):
        if env_id is not None and record["env_id"] != env_id:
            continue
        if task is not None and record["task"] != task:
            continue
        if scale is not None and record["scale"] != scale:
            continue
        if systems is not None and record["system_id"] not in systems:
            continue
        out.append(record)
    return out


def env_ids(records: Iterable[ScoreRecord]) -> List[str]:
    """Sorted unique ``env_id`` values present in *records*."""
    return sorted({record["env_id"] for record in records})


def run_ids(records: Iterable[ScoreRecord]) -> List[str]:
    """Sorted unique ``run_id`` values present in *records* (provenance, AGENTS.md §3.6)."""
    return sorted({record["run_id"] for record in records})


def system_ids(records: Iterable[ScoreRecord]) -> List[str]:
    """Sorted unique ``system_id`` values present in *records*."""
    return sorted({record["system_id"] for record in records})


def env_metadata(records: Iterable[ScoreRecord]) -> List[EnvMeta]:
    """Per-env metadata + parse failure rate (ADR-011 D7), sorted by ``env_id``.

    ``env_id`` is a hash of the whole environment cell (FR-001), so task / scale /
    judge fields must be constant within an env; a mismatch means the store mixes
    incompatible provenance and raises :class:`ValueError`.
    """
    fields = ("task", "scale", "benchmark", "judge_model", "judge_model_version")
    by_env: Dict[str, List[ScoreRecord]] = {}
    for record in sorted_records(records):
        by_env.setdefault(record["env_id"], []).append(record)

    out: List[EnvMeta] = []
    for env, env_records in sorted(by_env.items()):
        head = env_records[0]
        for field in fields:
            values = {str(r[field]) for r in env_records}  # type: ignore[literal-required]
            if len(values) > 1:
                raise ValueError(f"env_id={env} carries multiple {field} values: {sorted(values)}")
        temperatures = {r["temperature"] for r in env_records}
        if len(temperatures) > 1:
            raise ValueError(f"env_id={env} carries multiple temperature values")
        n_records = len(env_records)
        n_scored = sum(1 for r in env_records if r["parse_ok"] and r["parsed_score"] is not None)
        out.append(
            EnvMeta(
                env_id=env,
                task=head["task"],
                scale=head["scale"],
                benchmark=head["benchmark"],
                judge_model=head["judge_model"],
                judge_model_version=head["judge_model_version"],
                temperature=head["temperature"],
                n_records=n_records,
                n_scored=n_scored,
                parse_failure_rate=(n_records - n_scored) / n_records if n_records else 0.0,
            )
        )
    return out


def group_by_cell(
    records: Iterable[ScoreRecord],
    *,
    repeats: Optional[Sequence[int]] = None,
) -> Dict[CellKey, Dict[int, float]]:
    """Group scored records into ``{cell: {repeat_idx: score}}`` (insertion order = sorted order).

    *repeats*, when given, restricts the result to those ``repeat_idx`` values —
    this is how the ADR-007 screening / estimation split is enforced downstream.
    Duplicate call keys with conflicting scores raise :class:`DuplicateRepeatError`
    (the call key is idempotent by construction, FR-002).
    """
    allowed = set(repeats) if repeats is not None else None
    cells: Dict[CellKey, Dict[int, float]] = {}
    for record in scored_records(records):
        repeat_idx = record["repeat_idx"]
        if allowed is not None and repeat_idx not in allowed:
            continue
        key = CellKey(record["env_id"], record["item_id"], record["system_id"])
        score = record["parsed_score"]
        assert score is not None  # guaranteed by scored_records
        bucket = cells.setdefault(key, {})
        previous = bucket.get(repeat_idx)
        if previous is not None and previous != score:
            raise DuplicateRepeatError(
                f"call key {key}/repeat_idx={repeat_idx} has conflicting scores "
                f"{previous} and {score}"
            )
        bucket[repeat_idx] = score
    return cells


def cell_values(cell: Dict[int, float]) -> List[float]:
    """Scores of one cell ordered by ``repeat_idx`` (deterministic)."""
    return [cell[idx] for idx in sorted(cell)]


def cell_repeats(cell: Dict[int, float]) -> List[int]:
    """Sorted ``repeat_idx`` values available in one cell."""
    return sorted(cell)


def items_by_system(
    cells: Dict[CellKey, Dict[int, float]],
    env_id: str,
    system_id: str,
) -> List[str]:
    """Sorted item ids scored for ``(env_id, system_id)``."""
    return sorted(
        key.item_id for key in cells if key.env_id == env_id and key.system_id == system_id
    )


def records_digest(records: Iterable[ScoreRecord]) -> str:
    """SHA-256 over the canonical (call key, score, run_id) projection of *records*.

    Traceability handle for derived outputs: identical inputs ⇒ identical digest,
    independent of record order or of fields that do not affect an analysis.
    """
    projection = [
        [
            r["env_id"],
            r["item_id"],
            r["system_id"],
            r["repeat_idx"],
            r["run_id"],
            r["parsed_score"] if r["parse_ok"] else None,
        ]
        for r in sorted_records(records)
    ]
    payload = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
