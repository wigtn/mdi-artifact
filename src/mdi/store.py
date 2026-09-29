"""Append-only raw score store + immutable input snapshots (FR-003, PRD §5.2, ADR-009/006).

Invariants (AGENTS.md §3.1): ``data/raw/`` and ``data/inputs/`` are append-only
and written by the runner only — never by hand or ad-hoc script. Parsing bugs
are fixed by re-running the parser over raw records, never by editing stored
records.

Integrity notes (PRD §5.2): single-writer per shard (advisory file lock);
on load every line is validated, and a crash-truncated last line is quarantined
into a sidecar rather than failing the shard — only that one call is
re-executed (the idempotent call key means no double billing; RPO 0). A
corrupt line that is *not* the final line is an integrity error, not a
truncation, and raises.

Three concerns live here, all of them "the data layer":

1. score record schema v2 + append-only shard I/O + the resume index;
2. the content-addressed input snapshot store under ``data/inputs/``
   (ADR-009 §2) and ``request_payload_hash`` derivation (ADR-009 §1);
3. loading evaluation items (local JSONL, or a public dataset pinned to a
   content-addressed cache) that the runner materializes into the snapshot
   store before any judge call.
"""

import errno
import fcntl
import hashlib
import itertools
import json
import math
import os
import types
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Type,
    TypedDict,
    cast,
)

SCHEMA_VERSION: int = 2

DIGEST_PREFIX = "sha256:"


class Usage(TypedDict):
    """Token usage for a single judge call."""

    in_tokens: int
    out_tokens: int


class ScoreRecord(TypedDict):
    """One raw score record — PRD §5.2 schema v2 (provenance fields per ADR-009/006)."""

    schema_version: int
    run_id: str
    env_id: str
    judge_model: str
    judge_model_version: str
    judge_tier: str
    serving_engine: Optional[str]
    quantization: Optional[str]
    seed: int
    prompt_id: str
    paraphrase_id: str
    temperature: float
    scale: str
    task: str
    benchmark: str
    item_id: str
    system_id: str
    repeat_idx: int
    request_payload_hash: str
    raw_response: str
    parsed_score: Optional[float]
    parse_ok: bool
    usage: Usage
    cost_usd: float
    ts: str


REQUIRED_FIELDS: Set[str] = set(ScoreRecord.__annotations__)


class StoreIntegrityError(Exception):
    """A raw shard contains a corrupt line that is not a crash-truncated tail."""


class CallKey(NamedTuple):
    """The idempotent judge-call key (FR-002 / PRD §5.2)."""

    env_id: str
    item_id: str
    system_id: str
    repeat_idx: int


def call_key(record: ScoreRecord) -> CallKey:
    """Return the idempotent call key of *record*."""
    return CallKey(
        env_id=record["env_id"],
        item_id=record["item_id"],
        system_id=record["system_id"],
        repeat_idx=record["repeat_idx"],
    )


class ShardLoad(NamedTuple):
    """Result of loading one raw shard: valid records plus quarantined lines."""

    records: List[ScoreRecord]
    quarantined: List[Dict[str, Any]]


# --------------------------------------------------------------------------------------
# Raw shard I/O
# --------------------------------------------------------------------------------------


def shard_path(raw_dir: str, experiment: str, env_id: str) -> str:
    """Deterministic shard path for one environment: ``<raw_dir>/<experiment>/<env_id>.jsonl``."""
    return os.path.join(raw_dir, experiment, f"{env_id}.jsonl")


def quarantine_path(store_path: str) -> str:
    """Sidecar path holding lines quarantined from *store_path*."""
    return f"{store_path}.quarantine.jsonl"


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of *path* if it does not exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _validate_line(line: str) -> ScoreRecord:
    """Parse and validate one JSONL line into a :class:`ScoreRecord`; raise on any defect."""
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError("record is not a JSON object")
    missing = REQUIRED_FIELDS - set(obj)
    extra = set(obj) - REQUIRED_FIELDS
    if missing or extra:
        raise ValueError(f"field mismatch (missing={sorted(missing)}, extra={sorted(extra)})")
    if obj["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version {obj['schema_version']!r} != {SCHEMA_VERSION}")
    usage = obj["usage"]
    if not isinstance(usage, dict) or set(usage) != {"in_tokens", "out_tokens"}:
        raise ValueError("usage must carry exactly in_tokens and out_tokens")
    return cast(ScoreRecord, obj)


class ShardWriter:
    """Single-writer append handle for one raw shard.

    Holds an exclusive advisory lock (``flock``) for its lifetime, so a second
    writer on the same shard fails fast instead of interleaving lines. If the
    shard does not end with a newline (crash mid-write), a newline is appended
    first: the truncated bytes stay on disk verbatim (raw is immutable) and
    become a standalone line that :func:`load_shard` quarantines.
    """

    def __init__(self, store_path: str) -> None:
        """Prepare a writer for *store_path* (the file is opened on ``__enter__``)."""
        self.store_path = store_path
        self._fh: Optional[TextIO] = None

    def __enter__(self) -> "ShardWriter":
        """Open the shard, take the exclusive lock, and repair a truncated tail."""
        _ensure_parent_dir(self.store_path)
        fh = open(self.store_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:  # pragma: no cover - platform/contention dependent
            fh.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                raise StoreIntegrityError(
                    f"shard already locked by another writer: {self.store_path}"
                ) from exc
            raise
        self._fh = fh
        self._terminate_partial_line()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[types.TracebackType],
    ) -> None:
        """Release the lock and close the shard."""
        fh = self._fh
        self._fh = None
        if fh is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    def _terminate_partial_line(self) -> None:
        """Isolate a crash-truncated tail so later appends stay parseable.

        The truncated bytes are never deleted (raw is immutable): they are
        recorded in the quarantine sidecar and terminated with a newline, which
        turns them into a standalone line :func:`load_shard` knows to skip.
        """
        fh = self._require_fh()
        fh.seek(0, os.SEEK_END)
        if fh.tell() == 0:
            return
        with open(self.store_path, "rb") as probe:
            content = probe.read()
        if content.endswith(b"\n"):
            return
        fragment = content.rsplit(b"\n", 1)[-1].decode("utf-8", errors="replace")
        _write_quarantine(
            self.store_path,
            [
                {
                    "shard": self.store_path,
                    "line_no": content.count(b"\n") + 1,
                    "reason": "truncated_tail: repaired by writer before append",
                    "raw_line": fragment,
                }
            ],
        )
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())

    def _require_fh(self) -> TextIO:
        """Return the open file handle or raise if the writer is not active."""
        if self._fh is None:
            raise StoreIntegrityError("ShardWriter used outside its context manager")
        return self._fh

    def append(self, record: ScoreRecord) -> None:
        """Append one validated record and fsync it (PRD §4.4: RPO 0)."""
        fh = self._require_fh()
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        _validate_line(line)
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def append_record(store_path: str, record: ScoreRecord) -> None:
    """Append one score record to a raw store shard (JSONL).

    Convenience wrapper that opens, locks, writes, fsyncs and closes. Hot loops
    should hold a :class:`ShardWriter` open instead.
    """
    with ShardWriter(store_path) as writer:
        writer.append(record)


def load_shard(store_path: str, quarantine: bool = True) -> ShardLoad:
    """Load one shard with row-level validation (PRD §5.2).

    A defective **final** line is treated as a crash-truncated tail: it is
    quarantined (recorded in the sidecar when *quarantine* is true) and the rest
    of the shard loads normally. A defective non-final line is corruption and
    raises :class:`StoreIntegrityError`.
    """
    if not os.path.exists(store_path):
        return ShardLoad(records=[], quarantined=[])

    with open(store_path, "r", encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    # A well-formed shard ends with "\n", so the split leaves a trailing "".
    if lines and lines[-1] == "":
        lines.pop()

    known = _quarantined_lines(store_path)
    records: List[ScoreRecord] = []
    quarantined: List[Dict[str, Any]] = []
    last_index = len(lines) - 1
    for index, line in enumerate(lines):
        if line.strip() == "":
            continue
        try:
            records.append(_validate_line(line))
        except (ValueError, json.JSONDecodeError) as exc:
            if index != last_index and line not in known:
                raise StoreIntegrityError(
                    f"{store_path}:{index + 1}: corrupt record (not a truncated tail): {exc}"
                ) from exc
            quarantined.append(
                {
                    "shard": store_path,
                    "line_no": index + 1,
                    "reason": f"truncated_tail: {exc}",
                    "raw_line": line,
                }
            )

    if quarantine and quarantined:
        _write_quarantine(store_path, quarantined)
    return ShardLoad(records=records, quarantined=quarantined)


def _quarantine_entries(store_path: str) -> List[Dict[str, Any]]:
    """Read the shard's quarantine sidecar (empty when there is none)."""
    path = quarantine_path(store_path)
    if not os.path.exists(path):
        return []
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def _quarantined_lines(store_path: str) -> Set[str]:
    """Raw lines already accepted into quarantine — tolerated on every later load."""
    return {str(entry.get("raw_line", "")) for entry in _quarantine_entries(store_path)}


def _write_quarantine(store_path: str, entries: Sequence[Dict[str, Any]]) -> None:
    """Append quarantined lines to the shard's sidecar (idempotent per raw line)."""
    path = quarantine_path(store_path)
    existing = _quarantined_lines(store_path)
    _ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as fh:
        for entry in entries:
            raw_line = str(entry.get("raw_line", ""))
            if raw_line in existing:
                continue
            existing.add(raw_line)
            fh.write(
                json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
            )


def read_records(store_path: str) -> List[ScoreRecord]:
    """Read and validate all records from a raw store shard.

    Row-level validation on load; a crash-truncated final line is quarantined
    instead of failing the whole shard (PRD §5.2).
    """
    return load_shard(store_path).records


def resume_index(records: Iterable[ScoreRecord]) -> Set[CallKey]:
    """Set of already-billed call keys — a resumed run must not re-issue these (FR-002)."""
    return {call_key(record) for record in records}


def records_by_cell(
    records: Iterable[ScoreRecord],
) -> Dict[Tuple[str, str, str], List[ScoreRecord]]:
    """Group records by ``(env_id, item_id, system_id)``, each group sorted by ``repeat_idx``."""
    grouped: Dict[Tuple[str, str, str], List[ScoreRecord]] = {}
    for record in records:
        key = (record["env_id"], record["item_id"], record["system_id"])
        grouped.setdefault(key, []).append(record)
    for group in grouped.values():
        group.sort(key=lambda rec: rec["repeat_idx"])
    return grouped


# --------------------------------------------------------------------------------------
# Content-addressed input snapshot store (ADR-009)
# --------------------------------------------------------------------------------------


def content_digest(text: str) -> str:
    """SHA-256 of *text* as ``sha256:<hex>`` — the store's content address."""
    return DIGEST_PREFIX + hashlib.sha256(text.encode("utf-8")).hexdigest()


def request_payload_hash(rendered_prompt: str) -> str:
    """``request_payload_hash`` for a fully rendered judge prompt (ADR-009 §1)."""
    return content_digest(rendered_prompt)


def canonical_json(obj: Any) -> str:
    """Canonical JSON: sorted keys, compact separators, ASCII-safe."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def snapshot_path(inputs_dir: str, digest: str) -> str:
    """Path of the content-addressed snapshot for *digest* (``sha256:<hex>``)."""
    if not digest.startswith(DIGEST_PREFIX):
        raise ValueError(f"not a content digest: {digest!r}")
    hexdigest = digest[len(DIGEST_PREFIX) :]
    return os.path.join(inputs_dir, "sha256", hexdigest[:2], f"{hexdigest}.txt")


def snapshot_text(inputs_dir: str, text: str) -> str:
    """Write *text* into the immutable snapshot store; return its content digest.

    Write-once: an existing snapshot with the same digest is left untouched
    (content-addressed, so its bytes already equal *text*).
    """
    digest = content_digest(text)
    path = snapshot_path(inputs_dir, digest)
    if os.path.exists(path):
        return digest
    _ensure_parent_dir(path)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return digest


def snapshot_json(inputs_dir: str, obj: Any) -> str:
    """Snapshot *obj* as canonical JSON; return its content digest."""
    return snapshot_text(inputs_dir, canonical_json(obj))


def read_snapshot(inputs_dir: str, digest: str) -> str:
    """Read back a snapshot by content digest."""
    with open(snapshot_path(inputs_dir, digest), "r", encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------------------
# Evaluation items (the judge's inputs)
# --------------------------------------------------------------------------------------


class EvalItem(TypedDict):
    """One evaluation item: a source/instruction plus one output per system."""

    item_id: str
    source: str
    outputs: Dict[str, str]


class ItemSnapshot(TypedDict):
    """Provenance record for a materialized item set (written to ``data/inputs/``)."""

    inputs_digest: str
    item_digests: Dict[str, str]
    output_digests: Dict[str, Dict[str, str]]
    system_ids: List[str]


def load_items_jsonl(path: str) -> List[EvalItem]:
    """Load evaluation items from a local JSONL file, sorted by ``item_id``.

    Line schema::

        {"item_id": "...", "source": "...", "outputs": {"<system_id>": "...", ...}}

    Every item must carry the same system id set — otherwise the grid is ragged
    and pairwise comparisons are not well defined.
    """
    items: List[EvalItem] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if line.strip() == "":
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: item is not a JSON object")
            missing = {"item_id", "source", "outputs"} - set(obj)
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {sorted(missing)}")
            outputs = obj["outputs"]
            if not isinstance(outputs, dict) or not outputs:
                raise ValueError(f"{path}:{line_no}: outputs must be a non-empty object")
            items.append(
                EvalItem(
                    item_id=str(obj["item_id"]),
                    source=str(obj["source"]),
                    outputs={str(k): str(v) for k, v in outputs.items()},
                )
            )
    if not items:
        raise ValueError(f"{path}: no items")
    system_sets = {tuple(sorted(item["outputs"])) for item in items}
    if len(system_sets) != 1:
        raise ValueError(f"{path}: items disagree on the system id set: {sorted(system_sets)}")
    items.sort(key=lambda item: item["item_id"])
    return items


def system_ids(items: Sequence[EvalItem]) -> List[str]:
    """Sorted system ids shared by *items* (deterministic work ordering)."""
    return sorted(items[0]["outputs"])


def materialize_items(inputs_dir: str, items: Sequence[EvalItem]) -> ItemSnapshot:
    """Write every item source and system output into the snapshot store (ADR-009 §2).

    Returns the digest manifest, itself snapshotted, so any record can be traced
    back to the exact bytes that were judged.
    """
    item_digests: Dict[str, str] = {}
    output_digests: Dict[str, Dict[str, str]] = {}
    for item in items:
        item_digests[item["item_id"]] = snapshot_text(inputs_dir, item["source"])
        output_digests[item["item_id"]] = {
            sid: snapshot_text(inputs_dir, text) for sid, text in sorted(item["outputs"].items())
        }
    manifest = {
        "item_digests": item_digests,
        "output_digests": output_digests,
        "system_ids": system_ids(items),
    }
    inputs_digest = snapshot_json(inputs_dir, manifest)
    return ItemSnapshot(
        inputs_digest=inputs_digest,
        item_digests=item_digests,
        output_digests=output_digests,
        system_ids=system_ids(items),
    )


# --------------------------------------------------------------------------------------
# Item sources
# --------------------------------------------------------------------------------------
#
# Three source kinds, all resolved to the same ``List[EvalItem]``:
#
#   kind: jsonl               local file the user supplies (offline path)
#   kind: hf_datasets_server  a public HF dataset, fetched once through the
#                             read-only datasets-server API and pinned to a
#                             content-addressed cache under data/inputs/cache/.
#   kind: alpaca_eval_github  AlpacaEval 2.0 per-model outputs, fetched once as
#                             raw GitHub JSON and pinned to the same cache
#                             (see the source note above the builder below).
#
# The network is touched only when the cache for the exact source spec is cold;
# every later run (and every test) reads the pinned bytes. No judge API is
# involved here — this is input data, not scoring (AGENTS.md §3.2 unaffected).

HF_ROWS_URL = "https://datasets-server.huggingface.co/rows"

_SUMMEVAL_FACETS: Tuple[str, ...] = ("relevance", "coherence", "fluency", "consistency")


def source_cache_path(inputs_dir: str, spec: Dict[str, Any]) -> str:
    """Deterministic cache path for a remote item-source *spec*."""
    digest = hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()
    return os.path.join(inputs_dir, "cache", f"{digest}.json")


def _fetch_hf_rows(dataset: str, config: str, split: str, length: int, timeout: float) -> Any:
    """GET rows from the HF datasets-server (read-only, public datasets)."""
    import httpx  # imported lazily: the cached path must not require the network stack

    params: Dict[str, str] = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "offset": "0",
        "length": str(length),
    }
    response = httpx.get(HF_ROWS_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _load_source_payload(inputs_dir: str, spec: Dict[str, Any], allow_fetch: bool) -> Any:
    """Return the raw source payload for *spec*, fetching only on a cold cache."""
    cache_path = source_cache_path(inputs_dir, spec)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    if not allow_fetch:
        raise FileNotFoundError(
            f"item source cache is cold and fetching is disabled: {cache_path}. "
            "Run `uv run mdi estimate-cost --config <cfg>` once with network access, "
            "or switch the config to `inputs.kind: jsonl`."
        )
    payload = _fetch_hf_rows(
        dataset=str(spec["dataset"]),
        config=str(spec.get("config", "default")),
        split=str(spec.get("split", "test")),
        length=int(spec.get("fetch_rows", 100)),
        timeout=float(spec.get("timeout_s", 60.0)),
    )
    _ensure_parent_dir(cache_path)
    tmp = f"{cache_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(payload))
    os.replace(tmp, cache_path)
    return payload


def _closest_quality_pair(means: Sequence[float]) -> Tuple[int, int]:
    """Indices of the two systems with the smallest mean-human-score gap (ties: lowest index)."""
    best: Optional[Tuple[float, int, int]] = None
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            gap = abs(means[i] - means[j])
            candidate = (gap, i, j)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("need at least two systems to form a pair")
    return best[1], best[2]


def _closest_quality_cluster(means: Sequence[float], k: int) -> List[int]:
    """Indices of the *k* systems packed into the tightest mean-human-score window.

    Sorting by mean score makes the tightest set of *k* systems a contiguous
    window, so the smallest spread is found by scanning windows. Every pair drawn
    from the returned set is therefore a close pair, which is the regime FIP is
    about — and *k* systems give ``k*(k-1)/2`` pairs rather than one.

    The pair count matters beyond sample size: FIP's confidence interval comes
    from a cluster bootstrap over system pairs, and a single pair leaves nothing
    to resample (see :data:`mdi.stats.bootstrap.MIN_CLUSTERS_FOR_CI`). Two
    systems yield a point estimate with no interval; four yield six pairs.

    Ties break toward the lowest starting index, so the result is deterministic.
    """
    if k < 2:
        raise ValueError(f"need at least two systems, got k={k}")
    if len(means) < k:
        raise ValueError(f"source has {len(means)} systems, need {k}")
    order = sorted(range(len(means)), key=lambda index: (means[index], index))
    best_start = 0
    best_spread: Optional[float] = None
    for start in range(len(order) - k + 1):
        window = order[start : start + k]
        spread = means[window[-1]] - means[window[0]]
        if best_spread is None or spread < best_spread:
            best_spread, best_start = spread, start
    return sorted(order[best_start : best_start + k])


#: The SummEval release carries expert annotations on a 1--5 scale per facet, so
#: a *difference* of human scores normalizes to %p with this width, exactly like
#: a judge likert5 difference (ADR-001).
SUMMEVAL_HUMAN_WIDTH: float = 4.0

#: The item source every summarization config in this project points at. Pinned
#: here so the oracle analysis (ADR-029) can read the same snapshot the runner
#: scored without being handed a config — all four summarization configs share
#: `fetch_rows: 100` and therefore this exact cache entry.
#: Public alias of the annotation facets, for callers that need to name them
#: (the oracle analysis exposes them as a CLI flag for the ADR-029a weighting
#: sensitivity).
SUMMEVAL_FACETS: Tuple[str, ...] = _SUMMEVAL_FACETS

SUMMEVAL_SOURCE_SPEC: Dict[str, Any] = {
    "kind": "hf_datasets_server",
    "dataset": "mteb/summeval",
    "config": "default",
    "split": "test",
    "fetch_rows": 100,
}


def summeval_human_scores(
    payload: Any,
    *,
    facets: Sequence[str] = _SUMMEVAL_FACETS,
) -> Dict[str, Dict[str, float]]:
    """``{item_id: {system_id: expert score}}`` for every summarizer in the release.

    The SummEval rows carry, beside the machine summaries this project scores, the
    three-expert mean of each annotation facet. Averaging the facets with equal
    weight matches the judge prompt, which asks for overall quality "considering
    relevance, coherence, fluency, and factual consistency" — the same four axes.

    This is a *read* of the pinned snapshot (AGENTS.md §3.1): the annotations were
    already in the payload the runner fetched, and until now they were consulted
    only to pick close systems (:func:`_closest_quality_cluster`), never to check
    anything. System ids match :func:`_summeval_items` (``s_<index>`` over
    ``machine_summaries``), so the result indexes straight onto scored records.
    """
    if not facets:
        raise ValueError("need at least one annotation facet")
    out: Dict[str, Dict[str, float]] = {}
    for wrapper in payload["rows"]:
        row = wrapper["row"]
        missing = [facet for facet in facets if facet not in row]
        if missing:
            raise ValueError(f"SummEval payload has no annotation facet(s): {sorted(missing)}")
        n_systems = len(row["machine_summaries"])
        scores: Dict[str, float] = {}
        for index in range(n_systems):
            total = math.fsum(float(row[facet][index]) for facet in facets)
            scores[f"s_{index:02d}"] = total / len(facets)
        out[str(row["id"])] = scores
    return out


def read_source_cache(inputs_dir: str, spec: Dict[str, Any]) -> Any:
    """Read a pinned item-source payload for *spec*. Never fetches, never writes."""
    kind = str(spec.get("kind", "jsonl"))
    if kind != "hf_datasets_server":
        raise ValueError(f"read_source_cache supports hf_datasets_server, not {kind!r}")
    return _load_source_payload(inputs_dir, _hf_cache_spec(spec), allow_fetch=False)


def _hf_cache_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of an ``inputs:`` block that addresses the content cache."""
    return {
        "dataset": spec["dataset"],
        "config": spec.get("config", "default"),
        "split": spec.get("split", "test"),
        "fetch_rows": int(spec.get("fetch_rows", 100)),
    }


def _summeval_items(payload: Any, n_items: int, n_systems: int = 2) -> List[EvalItem]:
    """Build close-quality items from a SummEval datasets-server payload.

    Deterministic: rows are sorted by dataset id, the first *n_items* are kept,
    and the *n_systems* summarizers kept are those packed into the tightest
    window of mean human score (the four SummEval facets averaged) over those
    items. Spike-only selection rule — it does **not** implement ADR-007 (Exp 2
    held-out pair screening).
    """
    rows = [row["row"] for row in payload["rows"]]
    rows.sort(key=lambda row: str(row["id"]))
    rows = rows[:n_items]
    if len(rows) < n_items:
        raise ValueError(f"source has {len(rows)} rows, need {n_items}")

    selected = _closest_quality_cluster(_summeval_system_means(rows), n_systems)

    items: List[EvalItem] = []
    for row in rows:
        items.append(
            EvalItem(
                item_id=str(row["id"]),
                source=str(row["text"]),
                outputs={
                    f"s_{index:02d}": str(row["machine_summaries"][index]) for index in selected
                },
            )
        )
    items.sort(key=lambda item: item["item_id"])
    return items


def _summeval_system_means(rows: Sequence[Any]) -> List[float]:
    """Per-summarizer mean expert score over *rows* (the four facets, equal weight)."""
    available = len(rows[0]["machine_summaries"])
    means: List[float] = []
    for sys_idx in range(available):
        total = 0.0
        for row in rows:
            total += sum(float(row[facet][sys_idx]) for facet in _SUMMEVAL_FACETS) / len(
                _SUMMEVAL_FACETS
            )
        means.append(total / len(rows))
    return means


def gap_ladder_pairs(
    means: Sequence[float],
    targets: Sequence[float],
) -> List[Tuple[int, int]]:
    """One system pair per entry of *targets*, by external gap in %p.

    ADR-029 §e asks for a *known-effect ladder*: pairs whose expert-measured gap
    spans the reachable range, so detection can be read as a curve over a gap the
    judge did not produce. For each target this returns the pair whose gap is
    closest to it.

    Gaps are %p of the annotation scale (facets run 1--5, width 4), matching
    :func:`mdi.stats.oracle.oracle_gaps` — so a target reads in the same unit as
    every other %p figure in the project (ADR-001). The scale *offset* cancels in
    a difference, which is why only the width enters here.

    Deterministic: candidate pairs are generated in sorted index order and the
    key is ``(|gap - target|, low, high)``, so ties break toward the lowest
    indices. A target may select a pair another target already selected; the
    result keeps one entry per target and callers dedupe systems.
    """
    if len(means) < 2:
        raise ValueError(f"need at least two systems, got {len(means)}")
    if not targets:
        raise ValueError("need at least one ladder target")
    scale_width = 4.0
    pairs: List[Tuple[int, int]] = []
    for target in targets:
        best = min(
            itertools.combinations(range(len(means)), 2),
            key=lambda pair: (
                abs(abs(means[pair[0]] - means[pair[1]]) / scale_width * 100.0 - float(target)),
                pair[0],
                pair[1],
            ),
        )
        pairs.append(best)
    return pairs


def _summeval_ladder_items(
    payload: Any,
    n_items: int,
    targets: Sequence[float],
) -> List[EvalItem]:
    """Build known-effect-ladder items from a SummEval payload (ADR-029 §e).

    Same determinism contract as :func:`_summeval_items` — rows sorted by dataset
    id, first *n_items* kept — but the summarizers kept are the union of the pairs
    :func:`gap_ladder_pairs` picks for *targets*, rather than the tightest cluster.
    The two builders are deliberately separate: ``summeval_close_pair`` selects
    for *indistinguishability* (the FIP regime) and this one selects for *spread*,
    and no config should be able to get one while asking for the other.
    """
    rows = [row["row"] for row in payload["rows"]]
    rows.sort(key=lambda row: str(row["id"]))
    rows = rows[:n_items]
    if len(rows) < n_items:
        raise ValueError(f"source has {len(rows)} rows, need {n_items}")

    means = _summeval_system_means(rows)
    selected = sorted({index for pair in gap_ladder_pairs(means, targets) for index in pair})

    items: List[EvalItem] = []
    for row in rows:
        items.append(
            EvalItem(
                item_id=str(row["id"]),
                source=str(row["text"]),
                outputs={
                    f"s_{index:02d}": str(row["machine_summaries"][index]) for index in selected
                },
            )
        )
    items.sort(key=lambda item: item["item_id"])
    return items


# --- AlpacaEval 2.0 (instruction following) --------------------------------------------
#
# Public per-model outputs for the 805-instruction AlpacaEval 2.0 eval set: every
# released system answered the identical instruction list, which is exactly the
# "same items, multiple systems" grid this project needs.
#
# Source (verified 2026-08-02):
#   * Per-system outputs — one JSON array per system, row keys
#     {dataset, generator, instruction, output}, 805 rows each:
#       https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/<ref>/results/<system>/model_outputs.json
#     (~228 system directories under results/ as of 2026-08-02).
#   * Closeness signal — the published length-controlled (LC) win rates:
#       https://github.com/tatsu-lab/alpaca_eval/blob/main/docs/data_AlpacaEval_2/weighted_alpaca_eval_gpt4_turbo_leaderboard.csv
#     The CSV's ``name`` column does NOT reliably match the results/ directory
#     name (e.g. "Blendax.AI-gm-l6-vo31" vs directory "blendaxai-gm-l6-vo31"),
#     so machine-mapping leaderboard rows to output files is fragile. The
#     builder therefore takes an explicit ``systems`` list as the PRIMARY
#     mechanism, with a closest-cluster fallback over config-supplied
#     ``lc_win_rates`` copied from that CSV (system CHOICE stays with the first
#     author under ADR-002/ADR-007 — this loader is transport, not selection).
#   * License — code Apache-2.0, data CC BY-NC 4.0 (repo README badges,
#     https://github.com/tatsu-lab/alpaca_eval; HF mirror tatsu-lab/alpaca_eval
#     is tagged cc-by-nc-4.0).
#   * The HF datasets-server route used for SummEval is NOT available here:
#     ``tatsu-lab/alpaca_eval`` is a script-based dataset and the /rows API
#     answers "doesn't support this dataset" (checked 2026-08-02) — hence the
#     raw-GitHub fetch, pinned to the same cache directory.

ALPACAEVAL_RAW_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/{ref}/results/"
    "{system}/model_outputs.json"
)

ALPACAEVAL_DEFAULT_MAX_CHARS = 8000


def _alpacaeval_select_systems(spec: Dict[str, Any]) -> List[str]:
    """Resolve the AlpacaEval system list from config (pure, deterministic).

    Primary mechanism: an explicit ``systems`` list of ``results/`` directory
    names — the first author's recorded pick. Fallback: ``candidate_systems``
    plus ``lc_win_rates`` (length-controlled win rates copied from the published
    leaderboard CSV; see the source note above), from which the ``n_systems``
    candidates packed into the tightest win-rate window are selected via
    :func:`_closest_quality_cluster`. Selection never touches the network.
    """
    systems = spec.get("systems")
    if systems:
        selected = sorted(str(system) for system in systems)
        if len(set(selected)) != len(selected):
            raise ValueError("inputs.systems contains duplicates")
        n_systems = spec.get("n_systems")
        if n_systems is not None and int(n_systems) != len(selected):
            raise ValueError(
                f"inputs.n_systems={n_systems} disagrees with len(inputs.systems)={len(selected)}"
            )
        return selected
    candidates = spec.get("candidate_systems")
    win_rates = spec.get("lc_win_rates")
    if not candidates or not isinstance(win_rates, dict):
        raise ValueError(
            "alpacaeval builder needs inputs.systems (primary), or "
            "inputs.candidate_systems plus inputs.lc_win_rates (fallback; LC win "
            "rates copied from the published AlpacaEval 2.0 leaderboard CSV)"
        )
    names = sorted({str(candidate) for candidate in candidates})
    missing = [name for name in names if name not in win_rates]
    if missing:
        raise ValueError(f"inputs.lc_win_rates has no entry for: {missing}")
    means = [float(win_rates[name]) for name in names]
    selected_indices = _closest_quality_cluster(means, int(spec.get("n_systems", 4)))
    return [names[index] for index in selected_indices]


def _load_alpacaeval_payload(
    inputs_dir: str, systems: Sequence[str], spec: Dict[str, Any], allow_fetch: bool
) -> Dict[str, Any]:
    """Return ``{system: rows}`` for *systems*, fetched once and pinned to the cache.

    Mirrors :func:`_load_source_payload`: the raw GitHub files are fetched only
    when the cache for this exact (ref, system set) is cold; afterwards every
    run and every test reads the pinned bytes under ``data/inputs/cache/``.
    """
    ref = str(spec.get("ref", "main"))
    cache_spec: Dict[str, Any] = {
        "kind": "alpaca_eval_github",
        "ref": ref,
        "systems": sorted(str(system) for system in systems),
    }
    cache_path = source_cache_path(inputs_dir, cache_spec)
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            return cast(Dict[str, Any], json.load(fh))
    if not allow_fetch:
        raise FileNotFoundError(
            f"item source cache is cold and fetching is disabled: {cache_path}. "
            "Run `uv run mdi estimate-cost --config <cfg>` once with network access, "
            "or switch the config to `inputs.kind: jsonl`."
        )
    import urllib.parse

    import httpx  # imported lazily: the cached path must not require the network stack

    timeout = float(spec.get("timeout_s", 60.0))
    payload: Dict[str, Any] = {}
    for system in cache_spec["systems"]:
        url = ALPACAEVAL_RAW_URL_TEMPLATE.format(
            ref=urllib.parse.quote(ref, safe=""),
            system=urllib.parse.quote(system, safe=""),
        )
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        payload[system] = response.json()
    _ensure_parent_dir(cache_path)
    tmp = f"{cache_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(payload))
    os.replace(tmp, cache_path)
    return payload


def _alpacaeval_items(
    payload: Dict[str, Any],
    n_items: int,
    systems: Sequence[str],
    max_chars: int = ALPACAEVAL_DEFAULT_MAX_CHARS,
) -> List[EvalItem]:
    """Build instruction-following items from a cached AlpacaEval 2.0 payload.

    Deterministic: item ids are content-addressed (``ae2_`` plus the first 12
    hex chars of the instruction's SHA-256), items are sorted by id and the
    first *n_items* kept — so the subsample is stable across runs and across
    ``n_items`` values, and independent of file ordering. Every selected system
    must carry the identical instruction set (otherwise the grid is ragged and
    pairwise comparison is undefined).

    Truncation policy — WORKING ASSUMPTION, not an ADR decision: instructions
    and outputs are hard-truncated to their first *max_chars* characters, with
    no ellipsis marker (default :data:`ALPACAEVAL_DEFAULT_MAX_CHARS` = 8000,
    ~2k tokens at 4 chars/token). AlpacaEval outputs are long (median ~2.1k
    chars for gpt4_1106_preview; community entries far longer), and an
    unbounded tail would blow the judged-context and cost estimates. The cap
    keeps judged bytes deterministic and bounded; revisit alongside
    ADR-002/ADR-003 if it turns out to bite.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    selected = sorted(str(system) for system in systems)
    if not selected:
        raise ValueError("need at least one system")
    by_system: Dict[str, Dict[str, str]] = {}
    for system in selected:
        mapping: Dict[str, str] = {}
        for row in payload[system]:
            instruction = str(row["instruction"])
            if instruction in mapping:
                raise ValueError(f"duplicate instruction in outputs of {system!r}")
            mapping[instruction] = str(row["output"])
        by_system[system] = mapping
    shared = set(by_system[selected[0]])
    for system in selected[1:]:
        if set(by_system[system]) != shared:
            raise ValueError(
                f"systems disagree on the instruction set: {selected[0]!r} vs {system!r}"
            )
    if len(shared) < n_items:
        raise ValueError(f"source has {len(shared)} instructions, need {n_items}")

    items: List[EvalItem] = []
    for instruction in shared:
        digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:12]
        items.append(
            EvalItem(
                item_id=f"ae2_{digest}",
                source=instruction[:max_chars],
                outputs={system: by_system[system][instruction][:max_chars] for system in selected},
            )
        )
    if len({item["item_id"] for item in items}) != len(items):
        raise ValueError("item_id collision across AlpacaEval instructions")
    items.sort(key=lambda item: item["item_id"])
    return items[:n_items]


def resolve_items(
    spec: Dict[str, Any], inputs_dir: str, allow_fetch: bool = True
) -> List[EvalItem]:
    """Resolve an ``inputs:`` config block into evaluation items (deterministic)."""
    kind = str(spec.get("kind", "jsonl"))
    if kind == "jsonl":
        path = spec.get("path")
        if not path:
            raise ValueError("inputs.kind=jsonl requires inputs.path")
        items = load_items_jsonl(str(path))
        n_items = spec.get("n_items")
        if n_items is not None:
            items = items[: int(n_items)]
        return items
    if kind == "hf_datasets_server":
        # Config is validated BEFORE the payload load: a misspelled builder or a
        # ladder with no targets is a config error, and it should surface as one
        # rather than as a cold-cache fetch failure from a network call the run
        # was never going to be able to use.
        builder = str(spec.get("builder", "summeval_close_pair"))
        if builder not in ("summeval_close_pair", "summeval_gap_ladder"):
            raise ValueError(f"unknown inputs.builder: {builder!r}")
        targets = spec.get("ladder_targets")
        if builder == "summeval_gap_ladder" and not targets:
            raise ValueError("inputs.builder=summeval_gap_ladder requires inputs.ladder_targets")
        payload = _load_source_payload(inputs_dir, _hf_cache_spec(spec), allow_fetch=allow_fetch)
        if builder == "summeval_close_pair":
            return _summeval_items(
                payload,
                int(spec.get("n_items", 20)),
                int(spec.get("n_systems", 2)),
            )
        return _summeval_ladder_items(
            payload,
            int(spec.get("n_items", 100)),
            [float(target) for target in cast(Sequence[Any], targets)],
        )
    if kind == "alpaca_eval_github":
        builder = str(spec.get("builder", "alpacaeval_close_cluster"))
        if builder != "alpacaeval_close_cluster":
            raise ValueError(f"unknown inputs.builder: {builder!r}")
        systems = _alpacaeval_select_systems(spec)
        payload = _load_alpacaeval_payload(inputs_dir, systems, spec, allow_fetch=allow_fetch)
        return _alpacaeval_items(
            payload,
            int(spec.get("n_items", 100)),
            systems,
            int(spec.get("max_chars", ALPACAEVAL_DEFAULT_MAX_CHARS)),
        )
    raise ValueError(f"unknown inputs.kind: {kind!r}")


# --------------------------------------------------------------------------------------
# Regimes promotion loader (Exp 4 promotion-reversal case study — ADR-017)
# --------------------------------------------------------------------------------------
#
# Yohei Nakajima's "Regimes" self-improvement loop publishes, for every promoted
# transform, the held-out CONFIRM split scored before and after the transform —
# a paired binary vector over 100 questions. That is exactly the 2x2 McNemar
# input the Exp-4 promotion-reversal case study needs (ADR-017). This loader is
# read-only transport + parsing: no judge call, no scoring (AGENTS.md §3.2
# unaffected). It mirrors the SummEval / AlpacaEval cold-cache-then-read pattern
# above and pins each fetched report under data/inputs/cache/; every later run
# and every test reads the pinned bytes.
#
# Source (verified 2026-08-02):
#   * Repo:    https://github.com/yoheinakajima/regimes  (paper arXiv:2606.10241)
#   * License: Apache-2.0 (repo LICENSE, HTTP 200). We redistribute only derived
#     statistics; the raw per-question outcomes stay behind the pinned URL.
#   * Commit:  7ba11a9da4d7ebdb77e040b62efe905394d84187 (main HEAD @ 2026-06-08).
#   * Files:   results/run_seed{5,11,23,101}/report.json and
#              results/run_2026-05-31_seed7/report.json — seed 7's run lives
#              under a dated directory, the others do not.
#
# Actual JSON shape (confirmed against the fetched files — the data-availability
# note EXP4-regimes-data-check.md put two field names in the wrong block):
#   * report.json["promotions"][i] carries the paired vectors
#     confirm_baseline_outcomes and confirm_transform_outcomes (each a list of
#     100 records with question_id + correct) and the scalar confirm_delta.
#   * The named tallies confirm_n_recovered (= w->r gains, our b) and
#     confirm_n_introduced (= r->w losses, our c) live in the PARALLEL
#     report.json["attributions"][i] block, NOT on the promotion. The two blocks
#     agree 1:1 across all 14 promotions, so this loader derives b/c straight
#     from the paired vectors (the auditable 2x2) rather than trusting the
#     pre-tallied counts.

REGIMES_REPO: str = "https://github.com/yoheinakajima/regimes"
REGIMES_REF: str = "7ba11a9da4d7ebdb77e040b62efe905394d84187"
REGIMES_LICENSE: str = "Apache-2.0"
REGIMES_RAW_URL_TEMPLATE: str = (
    "https://raw.githubusercontent.com/yoheinakajima/regimes/{ref}/results/{run}/report.json"
)

#: Seed -> results/ directory segment. Seed 7's run is under a dated directory.
REGIMES_SEED_RUNS: Dict[int, str] = {
    5: "run_seed5",
    7: "run_2026-05-31_seed7",
    11: "run_seed11",
    23: "run_seed23",
    101: "run_seed101",
}

#: The five seeds of the reported multi-seed run (14 promotions in total).
REGIMES_DEFAULT_SEEDS: Tuple[int, ...] = (5, 7, 11, 23, 101)


class PromotionRecord(TypedDict):
    """One promoted transform's held-out CONFIRM paired evaluation (ADR-017).

    ``n_recovered`` (= b, wrong->right gains) and ``n_introduced`` (= c,
    right->wrong losses) are the discordant-pair counts of the McNemar 2x2,
    derived from the paired 100-question vectors. ``confirm_delta`` is the
    held-out accuracy change the loop's accept-if-better gate actually saw
    (equal to ``(b - c) / n_questions``). ``question_ids`` / ``base_correct`` /
    ``transform_correct`` are parallel and sorted by ``question_id``, so the
    record is a self-contained audit of the 2x2.
    """

    seed: int
    promo_idx: int
    name: str
    iteration_id: str
    confirm_delta: float
    n_recovered: int
    n_introduced: int
    n_discordant: int
    n_questions: int
    question_ids: List[str]
    base_correct: List[bool]
    transform_correct: List[bool]


def regimes_run_segment(seed: int) -> str:
    """The ``results/`` directory segment for one Regimes *seed* (raise if unknown)."""
    try:
        return REGIMES_SEED_RUNS[seed]
    except KeyError:
        raise ValueError(
            f"no Regimes results directory is known for seed {seed} "
            f"(known seeds: {sorted(REGIMES_SEED_RUNS)})"
        ) from None


def _regimes_cache_spec(seed: int, ref: str) -> Dict[str, Any]:
    """Content-address key for one pinned Regimes report."""
    return {"kind": "regimes_report", "ref": ref, "seed": int(seed)}


def _load_regimes_report(inputs_dir: str, seed: int, ref: str, allow_fetch: bool) -> Dict[str, Any]:
    """Return one seed's raw ``report.json``, fetched once and pinned to the cache.

    Mirrors :func:`_load_source_payload`: the raw GitHub file is fetched only
    when the cache for this exact (ref, seed) is cold; afterwards every run and
    every test reads the pinned bytes under ``data/inputs/cache/``.
    """
    cache_path = source_cache_path(inputs_dir, _regimes_cache_spec(seed, ref))
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            return cast(Dict[str, Any], json.load(fh))
    if not allow_fetch:
        raise FileNotFoundError(
            f"Regimes report cache is cold and fetching is disabled: {cache_path}. "
            "Run `uv run mdi analyze promotions --source regimes --exp exp4` once with "
            "network access to pin it."
        )
    import httpx  # imported lazily: the cached path must not require the network stack

    url = REGIMES_RAW_URL_TEMPLATE.format(ref=ref, run=regimes_run_segment(seed))
    response = httpx.get(url, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    payload = cast(Dict[str, Any], response.json())
    _ensure_parent_dir(cache_path)
    tmp = f"{cache_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(payload))
    os.replace(tmp, cache_path)
    return payload


def _outcome_map(outcomes: Any, where: str) -> Dict[str, bool]:
    """Map ``question_id -> correct`` from a list of outcome records; raise if malformed."""
    if not isinstance(outcomes, list):
        raise ValueError(f"{where}: expected a list of outcome records")
    mapping: Dict[str, bool] = {}
    for entry in outcomes:
        if not isinstance(entry, dict) or "question_id" not in entry or "correct" not in entry:
            raise ValueError(f"{where}: outcome record missing question_id/correct")
        qid = str(entry["question_id"])
        if qid in mapping:
            raise ValueError(f"{where}: duplicate question_id {qid!r}")
        mapping[qid] = bool(entry["correct"])
    return mapping


def parse_regimes_promotions(seed: int, report: Dict[str, Any]) -> List[PromotionRecord]:
    """Parse one seed's ``report.json`` into ordered promotion records (pure, deterministic).

    Raises :class:`ValueError` on a malformed report: no ``promotions`` list, a
    promotion missing the paired vectors or ``confirm_delta``, or baseline /
    transform vectors that disagree on the question-id set (pairing would be
    undefined). b/c are counted from the paired vectors, not from any pre-tallied
    ``confirm_n_*`` field.
    """
    promotions = report.get("promotions")
    if not isinstance(promotions, list):
        raise ValueError(f"seed {seed}: report has no 'promotions' list")
    records: List[PromotionRecord] = []
    for idx, promo in enumerate(promotions):
        where = f"seed {seed} promotions[{idx}]"
        if not isinstance(promo, dict):
            raise ValueError(f"{where}: not a JSON object")
        for field in ("confirm_baseline_outcomes", "confirm_transform_outcomes", "confirm_delta"):
            if field not in promo:
                raise ValueError(f"{where}: missing {field!r}")
        base = _outcome_map(
            promo["confirm_baseline_outcomes"], f"{where}.confirm_baseline_outcomes"
        )
        transform = _outcome_map(
            promo["confirm_transform_outcomes"], f"{where}.confirm_transform_outcomes"
        )
        if set(base) != set(transform):
            raise ValueError(
                f"{where}: baseline and transform outcomes disagree on the question-id set"
            )
        question_ids = sorted(base)
        n_recovered = sum(1 for q in question_ids if (not base[q]) and transform[q])
        n_introduced = sum(1 for q in question_ids if base[q] and (not transform[q]))
        records.append(
            PromotionRecord(
                seed=int(seed),
                promo_idx=idx,
                name=str(promo.get("name", "")),
                iteration_id=str(promo.get("iteration_id", "")),
                confirm_delta=float(promo["confirm_delta"]),
                n_recovered=n_recovered,
                n_introduced=n_introduced,
                n_discordant=n_recovered + n_introduced,
                n_questions=len(question_ids),
                question_ids=question_ids,
                base_correct=[base[q] for q in question_ids],
                transform_correct=[transform[q] for q in question_ids],
            )
        )
    return records


def load_regimes_promotions(
    inputs_dir: str,
    seeds: Sequence[int] = REGIMES_DEFAULT_SEEDS,
    *,
    ref: str = REGIMES_REF,
    allow_fetch: bool = True,
) -> Dict[int, List[PromotionRecord]]:
    """Load per-seed promotion records from the pinned Regimes reports (ADR-017).

    A cold cache with *allow_fetch* fetches each report once and pins it under
    ``data/inputs/cache/``; a cold cache without it raises. Returns
    ``{seed: [promotion, ...]}`` in each seed's promotion order, seeds ascending.
    """
    out: Dict[int, List[PromotionRecord]] = {}
    for seed in sorted({int(s) for s in seeds}):
        report = _load_regimes_report(inputs_dir, seed, ref, allow_fetch)
        out[seed] = parse_regimes_promotions(seed, report)
    return out
