"""Tests for raw-shard I/O, quarantine recovery, resume index and input snapshots."""

import json
import os
from typing import Any, Dict, List

import pytest

from conftest import write_items_jsonl
from mdi.stats.bootstrap import MIN_CLUSTERS_FOR_CI
from mdi.store import (
    REGIMES_REF,
    SCHEMA_VERSION,
    CallKey,
    ScoreRecord,
    ShardWriter,
    StoreIntegrityError,
    Usage,
    _alpacaeval_items,
    _alpacaeval_select_systems,
    _closest_quality_cluster,
    _regimes_cache_spec,
    append_record,
    canonical_json,
    content_digest,
    gap_ladder_pairs,
    load_items_jsonl,
    load_regimes_promotions,
    load_shard,
    materialize_items,
    parse_regimes_promotions,
    quarantine_path,
    read_records,
    read_snapshot,
    request_payload_hash,
    resolve_items,
    resume_index,
    snapshot_text,
    source_cache_path,
)


def make_record(repeat_idx: int, parse_ok: bool = True, item_id: str = "i_000") -> ScoreRecord:
    """Build a valid schema-v2 record for shard tests."""
    return ScoreRecord(
        schema_version=SCHEMA_VERSION,
        run_id="r_test",
        env_id="e_test",
        judge_model="test/judge-1",
        judge_model_version="test-version",
        judge_tier="api",
        serving_engine=None,
        quantization=None,
        seed=1000 + repeat_idx,
        prompt_id="p_test",
        paraphrase_id="pp_0",
        temperature=1.0,
        scale="likert5",
        task="summarization",
        benchmark="test_bench",
        item_id=item_id,
        system_id="s_A",
        repeat_idx=repeat_idx,
        request_payload_hash="sha256:deadbeef",
        raw_response='{"score": 4}',
        parsed_score=4.0 if parse_ok else None,
        parse_ok=parse_ok,
        usage=Usage(in_tokens=100, out_tokens=10),
        cost_usd=0.0001,
        ts="2026-07-31T00:00:00+00:00",
    )


def test_append_record_round_trips_when_shard_is_read_back(tmp_path: Any) -> None:
    """Appended records read back identically, in append order."""
    # Given: an empty shard path
    path = os.path.join(str(tmp_path), "raw", "exp", "e_test.jsonl")
    # When: two records are appended
    append_record(path, make_record(0))
    append_record(path, make_record(1))
    # Then: both come back intact
    records = read_records(path)
    assert [record["repeat_idx"] for record in records] == [0, 1]
    assert records[0]["usage"]["in_tokens"] == 100


def test_read_records_returns_empty_when_shard_does_not_exist(tmp_path: Any) -> None:
    """A missing shard is an empty store, not an error (first run)."""
    # Given / When / Then
    assert read_records(os.path.join(str(tmp_path), "nope.jsonl")) == []


def test_truncated_final_line_is_quarantined_and_shard_still_loads(tmp_path: Any) -> None:
    """A crash-truncated tail is quarantined; the intact prefix loads (PRD §5.2)."""
    # Given: a shard whose last line was cut off mid-write
    path = os.path.join(str(tmp_path), "e_test.jsonl")
    append_record(path, make_record(0))
    append_record(path, make_record(1))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"schema_version": 2, "run_id": "r_te')
    # When: the shard is loaded
    result = load_shard(path)
    # Then: the two complete records survive and the tail is quarantined, not lost
    assert [record["repeat_idx"] for record in result.records] == [0, 1]
    assert len(result.quarantined) == 1
    assert result.quarantined[0]["reason"].startswith("truncated_tail")
    with open(quarantine_path(path), "r", encoding="utf-8") as fh:
        sidecar = [json.loads(line) for line in fh if line.strip()]
    assert sidecar[0]["raw_line"].startswith('{"schema_version": 2')


def test_writer_terminates_truncated_tail_so_later_appends_stay_parseable(
    tmp_path: Any,
) -> None:
    """Recovery: a new writer isolates the truncated bytes instead of concatenating onto them."""
    # Given: a shard ending in a partial line
    path = os.path.join(str(tmp_path), "e_test.jsonl")
    append_record(path, make_record(0))
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"schema_version": 2, "run')
    # When: the runner reopens the shard and appends the re-executed call
    with ShardWriter(path) as writer:
        writer.append(make_record(1))
    # Then: the good records load and only the truncated fragment is quarantined
    result = load_shard(path)
    assert [record["repeat_idx"] for record in result.records] == [0, 1]
    assert len(result.quarantined) == 1


def test_load_shard_raises_when_a_non_final_line_is_corrupt(tmp_path: Any) -> None:
    """Corruption in the middle of a shard is an integrity error, not a truncation."""
    # Given: a shard with a bad line followed by a good one
    path = os.path.join(str(tmp_path), "e_test.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("not json at all\n")
        fh.write(json.dumps(make_record(0), sort_keys=True) + "\n")
    # When / Then: loading refuses to silently drop it
    with pytest.raises(StoreIntegrityError, match="not a truncated tail"):
        load_shard(path)


def test_append_rejects_record_when_schema_fields_are_wrong(tmp_path: Any) -> None:
    """A record that is not schema v2 never reaches the append-only store."""
    # Given: a record missing a required field
    path = os.path.join(str(tmp_path), "e_test.jsonl")
    bad: Dict[str, Any] = dict(make_record(0))
    del bad["cost_usd"]
    # When / Then: the writer rejects it
    with pytest.raises(ValueError, match="field mismatch"):
        append_record(path, bad)  # type: ignore[arg-type]


def test_second_writer_is_refused_while_a_shard_is_open(tmp_path: Any) -> None:
    """Single-writer per shard: a concurrent writer fails fast (PRD §5.2)."""
    # Given: an open writer on a shard
    path = os.path.join(str(tmp_path), "e_test.jsonl")
    with ShardWriter(path):
        # When / Then: a second writer cannot take the lock
        with pytest.raises(StoreIntegrityError, match="already locked"):
            with ShardWriter(path):
                pass


def test_resume_index_contains_every_completed_call_key(tmp_path: Any) -> None:
    """The resume index is keyed on (env_id, item_id, system_id, repeat_idx)."""
    # Given: a shard with two completed calls
    path = os.path.join(str(tmp_path), "e_test.jsonl")
    append_record(path, make_record(0))
    append_record(path, make_record(1, item_id="i_001"))
    # When: the resume index is built
    index = resume_index(read_records(path))
    # Then: both keys are present
    assert CallKey("e_test", "i_000", "s_A", 0) in index
    assert CallKey("e_test", "i_001", "s_A", 1) in index
    assert CallKey("e_test", "i_000", "s_A", 5) not in index


def test_snapshot_text_is_content_addressed_and_write_once(tmp_path: Any) -> None:
    """The input snapshot store addresses by SHA-256 and never rewrites (ADR-009 §2)."""
    # Given: an inputs directory
    inputs_dir = os.path.join(str(tmp_path), "inputs")
    # When: the same text is snapshotted twice
    first = snapshot_text(inputs_dir, "hello judge")
    second = snapshot_text(inputs_dir, "hello judge")
    # Then: one address, readable back verbatim
    assert first == second == content_digest("hello judge")
    assert read_snapshot(inputs_dir, first) == "hello judge"


def test_request_payload_hash_matches_the_rendered_prompt_digest() -> None:
    """request_payload_hash is the SHA-256 of the fully rendered prompt (ADR-009 §1)."""
    # Given / When / Then
    assert request_payload_hash("PROMPT") == content_digest("PROMPT")
    assert request_payload_hash("PROMPT").startswith("sha256:")


def test_materialize_items_snapshots_every_source_and_output(tmp_path: Any) -> None:
    """Every judged byte lands in the snapshot store before any call is made."""
    # Given: a local item file
    items_file = write_items_jsonl(os.path.join(str(tmp_path), "items.jsonl"), n_items=2)
    items = load_items_jsonl(items_file)
    inputs_dir = os.path.join(str(tmp_path), "inputs")
    # When: the runner materializes them
    snapshot = materialize_items(inputs_dir, items)
    # Then: sources and outputs are retrievable by digest, and the manifest is pinned
    assert snapshot["system_ids"] == ["s_A", "s_B"]
    assert read_snapshot(inputs_dir, snapshot["item_digests"]["i_000"]) == items[0]["source"]
    assert (
        read_snapshot(inputs_dir, snapshot["output_digests"]["i_000"]["s_A"])
        == items[0]["outputs"]["s_A"]
    )
    assert snapshot["inputs_digest"].startswith("sha256:")


def test_load_items_jsonl_raises_when_items_disagree_on_systems(tmp_path: Any) -> None:
    """A ragged item set is rejected — pairwise comparison would be undefined."""
    # Given: two items with different system id sets
    path = os.path.join(str(tmp_path), "ragged.jsonl")
    lines: List[Dict[str, Any]] = [
        {"item_id": "i_0", "source": "a", "outputs": {"s_A": "x", "s_B": "y"}},
        {"item_id": "i_1", "source": "b", "outputs": {"s_A": "x"}},
    ]
    with open(path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, sort_keys=True) + "\n")
    # When / Then
    with pytest.raises(ValueError, match="disagree on the system id set"):
        load_items_jsonl(path)


def test_closest_quality_cluster_picks_the_tightest_window() -> None:
    """The kept systems must be the k packed closest together, not the k best."""
    # Given: means whose tightest trio sits in the middle, away from the top scorer
    means = [1.0, 4.00, 4.05, 4.10, 9.0]
    # When: three systems are selected
    selected = _closest_quality_cluster(means, 3)
    # Then: the tight cluster wins, sorted, and the choice is deterministic
    assert selected == [1, 2, 3]
    assert selected == _closest_quality_cluster(means, 3)


def test_closest_quality_cluster_rejects_impossible_requests() -> None:
    """Fewer than a pair, or more systems than exist, must fail loudly."""
    # Given: three systems
    means = [1.0, 2.0, 3.0]
    # When / Then
    with pytest.raises(ValueError):
        _closest_quality_cluster(means, 1)
    with pytest.raises(ValueError):
        _closest_quality_cluster(means, 4)


def test_closest_quality_cluster_yields_enough_pairs_for_a_bootstrap() -> None:
    """Four systems must give six pairs — a single pair leaves the CI undefined."""
    # Given: five candidate systems
    means = [4.0, 4.1, 4.2, 4.3, 9.0]
    # When: four are selected
    selected = _closest_quality_cluster(means, 4)
    # Then: the pair count clears the cluster-bootstrap minimum
    n_pairs = len(selected) * (len(selected) - 1) // 2
    assert n_pairs == 6
    assert n_pairs >= MIN_CLUSTERS_FOR_CI


def test_gap_ladder_pairs_picks_the_pair_nearest_each_target() -> None:
    """Each target must select the pair whose %p gap is closest to it."""
    # Given: means one point apart, so gaps in %p are 25 x the index distance
    means = [1.0, 2.0, 3.0, 4.0, 5.0]
    # When: a ladder is asked for over reachable targets
    pairs = gap_ladder_pairs(means, [0, 25, 100])
    # Then: adjacent for 25, the extremes for 100, and the nearest-to-zero for 0
    assert pairs[1] == (0, 1)
    assert pairs[2] == (0, 4)
    assert abs(means[pairs[0][0]] - means[pairs[0][1]]) / 4.0 * 100.0 == 25.0


def test_gap_ladder_pairs_is_deterministic_under_ties() -> None:
    """Equal-gap candidates must resolve to the lowest indices, every call."""
    # Given: three systems whose every pair is the same distance apart
    means = [1.0, 1.0, 1.0]
    # When: the same target is requested twice
    first = gap_ladder_pairs(means, [0])
    second = gap_ladder_pairs(means, [0])
    # Then: the tie breaks toward the lowest indices and does not drift
    assert first == [(0, 1)]
    assert first == second


def test_gap_ladder_pairs_rejects_an_empty_ladder() -> None:
    """No targets, or nothing to pair, must fail loudly rather than score nothing."""
    # Given: a usable and an unusable set of means
    means = [1.0, 2.0]
    # When / Then
    with pytest.raises(ValueError, match="ladder target"):
        gap_ladder_pairs(means, [])
    with pytest.raises(ValueError, match="at least two systems"):
        gap_ladder_pairs([1.0], [0])


def test_gap_ladder_spans_wider_than_the_close_cluster() -> None:
    """The two builders must select for opposite things — spread vs indistinguishability."""
    # Given: a tight cluster plus two outliers
    means = [1.0, 4.00, 4.05, 4.10, 9.0]
    # When: each selection rule runs over the same systems
    cluster = _closest_quality_cluster(means, 3)
    # 200 %p is the widest gap available here (8.0 raw over a width-4 scale)
    ladder = sorted({index for pair in gap_ladder_pairs(means, [0, 200]) for index in pair})
    # Then: the ladder reaches the extremes the close cluster excludes
    cluster_spread = max(means[i] for i in cluster) - min(means[i] for i in cluster)
    ladder_spread = max(means[i] for i in ladder) - min(means[i] for i in ladder)
    assert ladder_spread > cluster_spread
    assert 0 in ladder and 4 in ladder
    assert 0 not in cluster and 4 not in cluster


def test_resolve_items_rejects_a_ladder_with_no_targets(tmp_path: Any) -> None:
    """A ladder config missing its targets must not silently fall back to close-pair."""
    # Given: a spec naming the ladder builder but carrying no targets
    spec = {
        "kind": "hf_datasets_server",
        "dataset": "mteb/summeval",
        "builder": "summeval_gap_ladder",
        "n_items": 1,
    }
    # When / Then
    with pytest.raises(ValueError, match="ladder_targets"):
        resolve_items(spec, str(tmp_path), allow_fetch=False)


def make_alpacaeval_payload(
    systems: List[str], n_instructions: int = 5, long_output_marker: str = ""
) -> Dict[str, Any]:
    """Fixture payload in the shape of the pinned AlpacaEval cache: {system: rows}.

    Rows are emitted in reverse order per system to prove the builder does not
    depend on file ordering.
    """
    instructions = [
        f"Instruction number {index}: explain topic {index}." for index in range(n_instructions)
    ]
    payload: Dict[str, Any] = {}
    for system in systems:
        rows = [
            {
                "dataset": "helpful_base",
                "generator": system,
                "instruction": instruction,
                "output": f"{system} answer to [{instruction}] {long_output_marker}",
            }
            for instruction in instructions
        ]
        payload[system] = list(reversed(rows))
    return payload


def test_alpacaeval_items_uses_explicit_systems_and_is_deterministic() -> None:
    """The explicit systems list is honored and repeated builds are identical."""
    # Given: a three-system payload and an explicit two-system selection
    payload = make_alpacaeval_payload(["m_alpha", "m_beta", "m_gamma"])
    # When: items are built twice for two of the systems
    first = _alpacaeval_items(payload, 3, ["m_beta", "m_alpha"])
    second = _alpacaeval_items(payload, 3, ["m_alpha", "m_beta"])
    # Then: same items, sorted by content-addressed id, only the requested systems
    assert first == second
    assert [item["item_id"] for item in first] == sorted(item["item_id"] for item in first)
    assert all(item["item_id"].startswith("ae2_") for item in first)
    assert all(sorted(item["outputs"]) == ["m_alpha", "m_beta"] for item in first)
    # And: a smaller n_items is a prefix of the larger build (stable subsample)
    assert _alpacaeval_items(payload, 2, ["m_alpha", "m_beta"]) == first[:2]


def test_alpacaeval_select_systems_prefers_explicit_list_over_fallback() -> None:
    """inputs.systems is the primary mechanism and must win over win-rate data."""
    # Given: a spec carrying both mechanisms
    spec: Dict[str, Any] = {
        "systems": ["m_beta", "m_alpha"],
        "candidate_systems": ["m_x", "m_y", "m_z"],
        "lc_win_rates": {"m_x": 10.0, "m_y": 11.0, "m_z": 30.0},
    }
    # When / Then: the explicit list is returned, sorted
    assert _alpacaeval_select_systems(spec) == ["m_alpha", "m_beta"]


def test_alpacaeval_select_systems_falls_back_to_closest_lc_cluster() -> None:
    """Without inputs.systems, the tightest LC-win-rate cluster is selected."""
    # Given: five candidates whose tightest trio is known
    spec: Dict[str, Any] = {
        "candidate_systems": ["m_a", "m_b", "m_c", "m_d", "m_e"],
        "lc_win_rates": {"m_a": 1.0, "m_b": 40.0, "m_c": 40.5, "m_d": 41.0, "m_e": 90.0},
        "n_systems": 3,
    }
    # When: systems are selected
    selected = _alpacaeval_select_systems(spec)
    # Then: the close cluster wins, deterministically
    assert selected == ["m_b", "m_c", "m_d"]
    assert selected == _alpacaeval_select_systems(spec)


def test_alpacaeval_select_systems_rejects_mismatched_n_systems() -> None:
    """n_systems disagreeing with the explicit list is a config error, not a guess."""
    # Given: two explicit systems but n_systems of four
    spec: Dict[str, Any] = {"systems": ["m_a", "m_b"], "n_systems": 4}
    # When / Then
    with pytest.raises(ValueError, match="disagrees"):
        _alpacaeval_select_systems(spec)


def test_alpacaeval_items_truncates_source_and_outputs_to_max_chars() -> None:
    """The working-assumption cap hard-truncates every judged text."""
    # Given: outputs padded far beyond the cap
    payload = make_alpacaeval_payload(["m_a", "m_b"], long_output_marker="x" * 500)
    # When: items are built with a small cap
    items = _alpacaeval_items(payload, 2, ["m_a", "m_b"], max_chars=64)
    # Then: no judged text exceeds the cap
    assert all(len(item["source"]) <= 64 for item in items)
    assert all(len(text) <= 64 for item in items for text in item["outputs"].values())
    assert any(len(text) == 64 for item in items for text in item["outputs"].values())


def test_alpacaeval_items_raises_when_systems_disagree_on_instructions() -> None:
    """A ragged instruction set must fail loudly — pairwise would be undefined."""
    # Given: one system missing an instruction
    payload = make_alpacaeval_payload(["m_a", "m_b"])
    payload["m_b"] = payload["m_b"][:-1]
    # When / Then
    with pytest.raises(ValueError, match="disagree on the instruction set"):
        _alpacaeval_items(payload, 2, ["m_a", "m_b"])


def test_resolve_items_dispatches_to_alpacaeval_builder_from_warm_cache(tmp_path: Any) -> None:
    """kind=alpaca_eval_github resolves offline once the cache is pinned."""
    # Given: a pinned cache for the exact (ref, systems) spec — no network involved
    inputs_dir = os.path.join(str(tmp_path), "inputs")
    payload = make_alpacaeval_payload(["m_a", "m_b"])
    cache_spec: Dict[str, Any] = {
        "kind": "alpaca_eval_github",
        "ref": "main",
        "systems": ["m_a", "m_b"],
    }
    cache_path = source_cache_path(inputs_dir, cache_spec)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(payload))
    spec: Dict[str, Any] = {
        "kind": "alpaca_eval_github",
        "builder": "alpacaeval_close_cluster",
        "ref": "main",
        "systems": ["m_b", "m_a"],
        "n_items": 3,
    }
    # When: items are resolved with fetching disabled
    items = resolve_items(spec, inputs_dir, allow_fetch=False)
    # Then: the builder ran from the pinned bytes
    assert len(items) == 3
    assert all(sorted(item["outputs"]) == ["m_a", "m_b"] for item in items)


def test_resolve_items_rejects_unknown_alpacaeval_builder(tmp_path: Any) -> None:
    """An unknown builder name under the AlpacaEval kind must not fall through."""
    # Given: a spec naming a builder that does not exist
    spec: Dict[str, Any] = {"kind": "alpaca_eval_github", "builder": "nope", "systems": ["m_a"]}
    # When / Then
    with pytest.raises(ValueError, match="unknown inputs.builder"):
        resolve_items(spec, os.path.join(str(tmp_path), "inputs"), allow_fetch=False)


def test_resolve_items_refuses_to_fetch_alpacaeval_when_cache_is_cold(tmp_path: Any) -> None:
    """A cold cache with fetching disabled is an explicit error, not a silent call."""
    # Given: an empty inputs dir
    spec: Dict[str, Any] = {"kind": "alpaca_eval_github", "systems": ["m_a", "m_b"]}
    # When / Then
    with pytest.raises(FileNotFoundError, match="cache is cold"):
        resolve_items(spec, os.path.join(str(tmp_path), "inputs"), allow_fetch=False)


# --- Regimes promotion loader (Exp 4 case study — ADR-017) ------------------------------


def make_regimes_report(promos: List[Any]) -> Dict[str, Any]:
    """Build a report.json-shaped dict whose promotions realize the given (b, c, delta)."""
    promotions: List[Dict[str, Any]] = []
    for b, c, delta in promos:
        base: List[Dict[str, Any]] = []
        transform: List[Dict[str, Any]] = []
        for index in range(100):
            qid = f"q{index:04d}"
            if index < b:  # wrong -> right
                base_ok, transform_ok = False, True
            elif index < b + c:  # right -> wrong
                base_ok, transform_ok = True, False
            else:  # concordant
                base_ok, transform_ok = True, True
            base.append({"question_id": qid, "correct": base_ok})
            transform.append({"question_id": qid, "correct": transform_ok})
        promotions.append(
            {
                "name": "llm_reader_prompt_transform",
                "iteration_id": "loop-001",
                "confirm_delta": delta,
                "confirm_baseline_outcomes": base,
                "confirm_transform_outcomes": transform,
            }
        )
    return {"promotions": promotions, "attributions": []}


def test_parse_regimes_promotions_counts_discordant_pairs_from_paired_vectors() -> None:
    """b (w->r) and c (r->w) are counted from the 100-question paired vectors."""
    # Given: a report with a lopsided and a symmetric promotion
    report = make_regimes_report([(11, 1, 0.10), (7, 6, 0.01)])
    # When: it is parsed
    records = parse_regimes_promotions(5, report)
    # Then: the discordant counts and metadata come out intact and ordered
    assert [(r["n_recovered"], r["n_introduced"]) for r in records] == [(11, 1), (7, 6)]
    assert [r["promo_idx"] for r in records] == [0, 1]
    assert records[0]["seed"] == 5
    assert records[0]["n_discordant"] == 12
    assert records[0]["n_questions"] == 100
    assert records[0]["confirm_delta"] == pytest.approx(0.10)


def test_parse_regimes_promotions_rejects_a_report_without_promotions() -> None:
    """A report missing the promotions list is malformed, not empty."""
    # Given / When / Then
    with pytest.raises(ValueError, match="no 'promotions' list"):
        parse_regimes_promotions(5, {"attributions": []})


def test_parse_regimes_promotions_rejects_mismatched_question_sets() -> None:
    """Baseline and transform vectors must pair on the same question ids."""
    # Given: a promotion whose transform vector drops a question id
    report = make_regimes_report([(3, 1, 0.02)])
    report["promotions"][0]["confirm_transform_outcomes"].pop()
    # When / Then: the pairing is refused loudly
    with pytest.raises(ValueError, match="disagree on the question-id set"):
        parse_regimes_promotions(5, report)


def test_parse_regimes_promotions_rejects_a_promotion_missing_confirm_delta() -> None:
    """A promotion without confirm_delta cannot be scored and must raise."""
    # Given: a promotion stripped of its delta
    report = make_regimes_report([(5, 2, 0.03)])
    del report["promotions"][0]["confirm_delta"]
    # When / Then
    with pytest.raises(ValueError, match="confirm_delta"):
        parse_regimes_promotions(5, report)


def test_load_regimes_promotions_reads_a_warm_cache_offline(tmp_path: Any) -> None:
    """Once a seed's report is pinned, the loader resolves it with no network."""
    # Given: a report pinned under the content-addressed cache for seed 5
    inputs_dir = os.path.join(str(tmp_path), "inputs")
    report = make_regimes_report([(8, 0, 0.08), (11, 1, 0.10)])
    cache_path = source_cache_path(inputs_dir, _regimes_cache_spec(5, REGIMES_REF))
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(report))
    # When: the loader runs with fetching disabled for just that seed
    by_seed = load_regimes_promotions(inputs_dir, [5], allow_fetch=False)
    # Then: the pinned bytes are parsed into ordered promotion records
    assert sorted(by_seed) == [5]
    assert [(r["n_recovered"], r["n_introduced"]) for r in by_seed[5]] == [(8, 0), (11, 1)]


def test_load_regimes_promotions_refuses_a_cold_cache(tmp_path: Any) -> None:
    """A cold cache with fetching disabled is an explicit error, not a silent call."""
    # Given: an empty inputs dir
    # When / Then
    with pytest.raises(FileNotFoundError, match="cache is cold"):
        load_regimes_promotions(os.path.join(str(tmp_path), "inputs"), [5], allow_fetch=False)
