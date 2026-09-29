"""Tests for the cost-gated, resume-safe judging runner (FR-002/FR-004).

Every test injects a mock judge client — no test issues a network call, and the
call counter on the mock is what proves "resume does not re-bill".
"""

import json
import os
from typing import Any, Dict, List, Set

import pytest

from conftest import MockJudgeClient, write_items_jsonl
from mdi import runner
from mdi.cost import ledger_total_usd, read_ledger
from mdi.parse import parse_failure_rate
from mdi.stats.decay import grouped_cells
from mdi.store import ScoreRecord, load_shard, resume_index, shard_path

COST_PER_CALL = (100 * 1.0 + 10 * 2.0) / 1e6


def open_gate(config: Dict[str, Any], data_dir: str) -> None:
    """Stand in for `mdi estimate-cost`: write the receipt this config hash needs."""
    paths = runner.paths_for(data_dir)
    report = runner.estimate(config, pilot=True, data_dir=data_dir)
    runner.write_estimate_receipt(paths, runner.config_hash(config), report)


def all_records(config: Dict[str, Any], data_dir: str) -> List[ScoreRecord]:
    """Every record in every shard of the experiment, sorted deterministically."""
    plan = runner.plan_run(config, pilot=False, data_dir=data_dir)
    paths = runner.paths_for(data_dir)
    records: List[ScoreRecord] = []
    for env in sorted({cell.env_id for cell in plan.cells}):
        path = shard_path(paths.raw_dir, plan.experiment, env)
        records.extend(load_shard(path).records)
    records.sort(key=lambda r: (r["env_id"], r["item_id"], r["system_id"], r["repeat_idx"]))
    return records


# --------------------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------------------


def test_run_is_blocked_when_no_cost_estimate_is_on_record(
    config: Dict[str, Any], data_dir: str
) -> None:
    """No paid call without a cost gate (AGENTS.md §3.2): the receipt is mandatory."""
    # Given: a config that has never been through `mdi estimate-cost`
    client = MockJudgeClient()
    # When / Then: the pilot run refuses to start
    with pytest.raises(runner.EstimateRequired, match="no cost-gate receipt"):
        runner.run(config, pilot=True, data_dir=data_dir, client=client)
    assert client.calls == []


def test_full_run_is_blocked_when_the_pilot_flag_is_missing(
    config: Dict[str, Any], data_dir: str
) -> None:
    """Pilot before full grid (PRD §4.2): a full run without the pilot flag raises."""
    # Given: an estimated config with no completed pilot
    open_gate(config, data_dir)
    client = MockJudgeClient()
    # When / Then: the full run is refused before any call is issued
    with pytest.raises(runner.PilotRequired, match="no completed pilot"):
        runner.run(config, pilot=False, data_dir=data_dir, client=client)
    assert client.calls == []


def test_run_is_blocked_when_projected_cost_exceeds_the_cap(
    config: Dict[str, Any], data_dir: str
) -> None:
    """The cap blocks the run and no judge call is made (FR-004)."""
    # Given: a cap far below the projected worst-case spend
    config["budget"] = {"pilot_cap_usd": 1e-9, "cumulative_cap_usd": 1e-9}
    open_gate(config, data_dir)
    client = MockJudgeClient()
    # When / Then: the runner blocks with the cap in the message
    with pytest.raises(runner.BudgetExceeded, match="exceeds the pilot cap"):
        runner.run(config, pilot=True, data_dir=data_dir, client=client)
    assert client.calls == []
    assert read_ledger(runner.paths_for(data_dir).ledger_path) == []


def test_estimate_reports_block_when_the_cap_is_too_low(
    config: Dict[str, Any], data_dir: str
) -> None:
    """The dry-run estimator itself returns BLOCK, before anything is executed."""
    # Given: an unaffordable cap
    config["budget"] = {"pilot_cap_usd": 1e-9, "cumulative_cap_usd": 1e-9}
    # When: the estimate is computed
    report = runner.estimate(config, pilot=True, data_dir=data_dir)
    # Then: the verdict is BLOCK and the worst case is the 2N bound
    assert report["verdict"]["verdict"] == "BLOCK"
    assert report["worst_case"]["calls"] == 2 * report["expected"]["calls"]


# --------------------------------------------------------------------------------------
# Idempotence / resume
# --------------------------------------------------------------------------------------


def test_pilot_run_issues_exactly_the_planned_calls_when_first_executed(
    config: Dict[str, Any], data_dir: str
) -> None:
    """A fresh pilot issues items x systems x N calls and records each one."""
    # Given: an estimated config (2 pilot items x 2 systems x N=2)
    open_gate(config, data_dir)
    client = MockJudgeClient()
    # When: the pilot runs
    runner.run(config, pilot=True, data_dir=data_dir, client=client)
    # Then: eight calls, eight records, all parsed
    assert len(client.calls) == 8
    records = all_records(config, data_dir)
    assert len(records) == 8
    assert all(record["parse_ok"] for record in records)


def test_resume_skips_completed_keys_and_issues_no_new_calls(
    config: Dict[str, Any], data_dir: str
) -> None:
    """Re-running a completed slice re-bills nothing (FR-002, PRD §2.2)."""
    # Given: a completed pilot
    open_gate(config, data_dir)
    first = MockJudgeClient()
    run_id = runner.run(config, pilot=True, data_dir=data_dir, client=first)
    keys_before = resume_index(all_records(config, data_dir))
    # When: the same slice is resumed
    second = MockJudgeClient()
    runner.run(config, resume=run_id, pilot=True, data_dir=data_dir, client=second)
    # Then: zero calls issued and the store is unchanged
    assert second.calls == []
    assert resume_index(all_records(config, data_dir)) == keys_before
    assert len(all_records(config, data_dir)) == 8


def test_full_run_reuses_pilot_records_instead_of_re_billing_them(
    config: Dict[str, Any], data_dir: str
) -> None:
    """The pilot slice is a subset of the full slice — its calls are never repeated."""
    # Given: a completed pilot (2 items x 2 systems x N=2 = 8 records)
    open_gate(config, data_dir)
    runner.run(config, pilot=True, data_dir=data_dir, client=MockJudgeClient())
    # When: the full slice runs (3 items x 2 systems x N=4 = 24 target records)
    full = MockJudgeClient()
    runner.run(config, pilot=False, data_dir=data_dir, client=full)
    # Then: only the 16 missing repeats are billed
    assert len(full.calls) == 16
    assert len(all_records(config, data_dir)) == 24


def test_repeat_indices_are_contiguous_per_cell_after_a_full_run(
    config: Dict[str, Any], data_dir: str
) -> None:
    """Every cell ends with exactly N repeats indexed 0..N-1 (no gaps, no duplicates)."""
    # Given: pilot then full
    open_gate(config, data_dir)
    runner.run(config, pilot=True, data_dir=data_dir, client=MockJudgeClient())
    runner.run(config, pilot=False, data_dir=data_dir, client=MockJudgeClient())
    # When: records are grouped by cell
    per_cell: Dict[Any, List[int]] = {}
    for record in all_records(config, data_dir):
        key = (record["env_id"], record["item_id"], record["system_id"])
        per_cell.setdefault(key, []).append(record["repeat_idx"])
    # Then: each cell has 0..3
    assert len(per_cell) == 6
    for indices in per_cell.values():
        assert sorted(indices) == [0, 1, 2, 3]


# --------------------------------------------------------------------------------------
# Parse-failure redraw (ADR-011 D7)
# --------------------------------------------------------------------------------------


def single_cell_config(config: Dict[str, Any], tmp_path: Any) -> Dict[str, Any]:
    """Narrow the config to one item and one system so redraw counts are exact."""
    path = write_items_jsonl(os.path.join(str(tmp_path), "one.jsonl"), n_items=1, systems=("s_A",))
    config["inputs"] = {"kind": "jsonl", "path": path}
    config["repeats"] = {"n": 2, "max_attempts_factor": 2, "seed_base": 1000}
    config["pilot"] = {"n_items": 1, "n_repeats": 2}
    return config


def test_parse_failure_keeps_the_record_and_redraws_a_fresh_repeat_idx(
    config: Dict[str, Any], data_dir: str, tmp_path: Any
) -> None:
    """Failed parses are preserved and a new repeat_idx is drawn — never re-prompted."""
    # Given: a judge whose first two responses are unparseable
    config = single_cell_config(config, tmp_path)
    open_gate(config, data_dir)
    client = MockJudgeClient(responses=["I cannot rate this.", "no score here"])
    # When: the pilot runs (target N=2 valid repeats)
    runner.run(config, pilot=True, data_dir=data_dir, client=client)
    # Then: four records exist — two failures kept, two fresh repeats parsed
    records = all_records(config, data_dir)
    assert len(client.calls) == 4
    assert [record["repeat_idx"] for record in records] == [0, 1, 2, 3]
    assert [record["parse_ok"] for record in records] == [False, False, True, True]
    assert [record["parsed_score"] for record in records] == [None, None, 4.0, 4.0]
    assert [record["seed"] for record in records] == [1000, 1001, 1002, 1003]


def test_redraw_stops_at_the_two_n_attempt_cap_when_every_parse_fails(
    config: Dict[str, Any], data_dir: str, tmp_path: Any
) -> None:
    """The redraw loop terminates at 2N attempts (ADR-011 D7), not in an infinite loop."""
    # Given: a judge that never produces a parseable score
    config = single_cell_config(config, tmp_path)
    open_gate(config, data_dir)
    client = MockJudgeClient(default_response="I refuse to rate this summary.")
    # When: the pilot runs with target N=2
    runner.run(config, pilot=True, data_dir=data_dir, client=client)
    # Then: exactly 2N = 4 attempts were made, all preserved as failures
    assert len(client.calls) == 4
    records = all_records(config, data_dir)
    assert len(records) == 4
    assert not any(record["parse_ok"] for record in records)
    # And: the per-env failure rate is computable and equals 1.0
    stats = parse_failure_rate(records)
    assert len(stats) == 1
    assert stats[0].rate == 1.0


def test_capped_cell_is_not_retried_when_the_run_is_repeated(
    config: Dict[str, Any], data_dir: str, tmp_path: Any
) -> None:
    """A cell that exhausted its 2N budget is not re-billed on a later run."""
    # Given: a cell already at the attempt cap with zero valid repeats
    config = single_cell_config(config, tmp_path)
    open_gate(config, data_dir)
    runner.run(
        config,
        pilot=True,
        data_dir=data_dir,
        client=MockJudgeClient(default_response="no score"),
    )
    # When: the same slice is run again
    second = MockJudgeClient()
    runner.run(config, pilot=True, data_dir=data_dir, client=second)
    # Then: nothing further is issued
    assert second.calls == []
    assert len(all_records(config, data_dir)) == 4


# --------------------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------------------


def test_ledger_accumulates_actual_spend_across_runs(config: Dict[str, Any], data_dir: str) -> None:
    """Each run appends its realized spend; the total is the sum (FR-004)."""
    # Given: an estimated config
    open_gate(config, data_dir)
    ledger_path = runner.paths_for(data_dir).ledger_path
    # When: a pilot then a full run complete
    runner.run(config, pilot=True, data_dir=data_dir, client=MockJudgeClient())
    after_pilot = ledger_total_usd(ledger_path)
    runner.run(config, pilot=False, data_dir=data_dir, client=MockJudgeClient())
    after_full = ledger_total_usd(ledger_path)
    # Then: the ledger holds two entries whose costs match the priced call counts
    entries = read_ledger(ledger_path)
    assert [entry["calls"] for entry in entries] == [8, 16]
    assert after_pilot == pytest.approx(8 * COST_PER_CALL)
    assert after_full == pytest.approx(24 * COST_PER_CALL)
    assert entries[0]["pilot"] is True
    assert entries[1]["pilot"] is False
    assert all(entry["inputs_digest"].startswith("sha256:") for entry in entries)


def test_ledger_is_not_written_when_the_run_is_blocked(
    config: Dict[str, Any], data_dir: str
) -> None:
    """A blocked run spends nothing and records nothing."""
    # Given: a config with an unaffordable cap
    config["budget"] = {"pilot_cap_usd": 1e-9, "cumulative_cap_usd": 1e-9}
    open_gate(config, data_dir)
    # When / Then
    with pytest.raises(runner.BudgetExceeded):
        runner.run(config, pilot=True, data_dir=data_dir, client=MockJudgeClient())
    assert ledger_total_usd(runner.paths_for(data_dir).ledger_path) == 0.0


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def test_records_carry_full_schema_v2_provenance_after_a_run(
    config: Dict[str, Any], data_dir: str
) -> None:
    """Every record carries the ADR-009/006 provenance fields the audit needs."""
    # Given: a completed pilot
    open_gate(config, data_dir)
    runner.run(config, pilot=True, data_dir=data_dir, client=MockJudgeClient())
    # When: one record is inspected
    record = all_records(config, data_dir)[0]
    # Then: the request payload hash, model version echo and env metadata are present
    assert record["schema_version"] == 2
    assert record["request_payload_hash"].startswith("sha256:")
    assert record["judge_model_version"] == "test/judge-1-2026-07-31"
    assert record["scale"] == "likert5"
    assert record["temperature"] == 1.0
    assert record["cost_usd"] == pytest.approx(COST_PER_CALL)


def test_env_id_changes_when_the_prompt_template_text_changes(
    config: Dict[str, Any], data_dir: str
) -> None:
    """ADR-009 §3: the template source text, not its id, is part of env_id."""
    # Given: two configs identical except for template wording
    before = runner.plan_run(config, pilot=True, data_dir=data_dir).cells[0].env_id
    config["prompt"]["templates"]["likert5"] = (
        config["prompt"]["templates"]["likert5"] + "\nBe strict.\n"
    )
    # When: the grid is re-expanded under the same prompt_id
    after = runner.plan_run(config, pilot=True, data_dir=data_dir).cells[0].env_id
    # Then: the environment id differs
    assert before != after


def test_plan_is_deterministic_when_expanded_twice(config: Dict[str, Any], data_dir: str) -> None:
    """Work-item ordering is stable across expansions (AGENTS.md §3.3)."""
    # Given / When: the same config is planned twice
    first = runner.plan_run(config, pilot=False, data_dir=data_dir)
    second = runner.plan_run(config, pilot=False, data_dir=data_dir)
    # Then: identical cells, order included
    assert [(c.env_id, c.item_id, c.system_id) for c in first.cells] == [
        (c.env_id, c.item_id, c.system_id) for c in second.cells
    ]
    assert first.config_hash == second.config_hash
    assert first.inputs_digest == second.inputs_digest


def test_estimate_receipt_is_written_for_the_config_hash(
    config: Dict[str, Any], data_dir: str
) -> None:
    """The gate receipt is keyed by the canonical config hash, so edits reopen the gate."""
    # Given: an estimated config
    open_gate(config, data_dir)
    paths = runner.paths_for(data_dir)
    path = runner.estimate_receipt_path(paths, runner.config_hash(config))
    # When: the receipt is read back
    with open(path, "r", encoding="utf-8") as fh:
        receipt: Dict[str, Any] = json.load(fh)
    # Then: it names this config hash
    assert receipt["config_hash"] == runner.config_hash(config)
    # And: changing the config invalidates it
    config["repeats"]["n"] = 99
    assert not os.path.exists(runner.estimate_receipt_path(paths, runner.config_hash(config)))


def test_expected_estimate_uses_measured_output_not_the_cap(
    config: Dict[str, Any], data_dir: str
) -> None:
    """A high safety cap must not inflate the expected column (2026-08-02 recalibration)."""
    # Given: an output cap of 160 with a measured typical output of 48
    config["runtime"]["max_output_tokens"] = 160
    config["runtime"]["expected_output_tokens"] = 48
    plan = runner.plan_run(config, pilot=True, data_dir=data_dir)
    # When / Then: expected prices the measured size, worst prices the cap
    assert all(c["out_tokens"] == 48 for c in plan.expected_calls)
    assert all(c["out_tokens"] == 160 for c in plan.worst_case_calls)


def test_expected_output_defaults_to_the_cap_when_unset(
    config: Dict[str, Any], data_dir: str
) -> None:
    """Without a measured value the estimate stays conservative (cap = expected)."""
    # Given: only the cap is set
    config["runtime"]["max_output_tokens"] = 64
    config["runtime"].pop("expected_output_tokens", None)
    plan = runner.plan_run(config, pilot=True, data_dir=data_dir)
    # When / Then: both columns price the cap
    assert all(c["out_tokens"] == 64 for c in plan.expected_calls)
    assert all(c["out_tokens"] == 64 for c in plan.worst_case_calls)


# --------------------------------------------------------------------------------------
# Paraphrase family (Exp 3 procedural noise floor omega^2 — ADR-011 D1)
# --------------------------------------------------------------------------------------
#
# One environment spans several prompt paraphrases so the between-paraphrase
# (procedural) variance floor omega^2 is measurable within a single env_id. The
# four templates below keep the rubric CRITERIA identical (they all ask for a
# 1-5 score) and vary only the surface wording — the ADR-011 D1 paraphrase rule.

FAMILY_PARAPHRASE_IDS = ("pp_0", "pp_1", "pp_2", "pp_3")

FAMILY_TEMPLATES: Dict[str, str] = {
    "pp_0": (
        "Rate the summary 1-5 (form A).\n\nSource:\n{{source}}\n\n"
        'Summary:\n{{candidate}}\n\nReply {"score": <1-5>}.\n'
    ),
    "pp_1": (
        "Judge this summary from 1 to 5 (form B).\n\nArticle:\n{{source}}\n\n"
        'Summary:\n{{candidate}}\n\nReturn {"score": <1-5>}.\n'
    ),
    "pp_2": (
        "Give an overall 1-5 quality score (form C).\n\nOriginal:\n{{source}}\n\n"
        'Candidate:\n{{candidate}}\n\nRespond {"score": <1-5>}.\n'
    ),
    "pp_3": (
        "Score summary quality 1-5 (form D).\n\nText:\n{{source}}\n\n"
        'Proposed:\n{{candidate}}\n\nAnswer {"score": <1-5>}.\n'
    ),
}


def family_config(
    config: Dict[str, Any], n_paraphrases: int = 4, n_repeats: int = 8
) -> Dict[str, Any]:
    """Turn the offline base config into a paraphrase-family (Exp 3) config.

    pp_0 inherits the base template; pp_1.. each override likert5 with a distinct
    reworded rubric. The pilot slice carries the full *n_repeats* so a pilot run
    exercises the whole round-robin without needing the full-grid pilot flag.
    """
    ids = list(FAMILY_PARAPHRASE_IDS[:n_paraphrases])
    paraphrases: List[Dict[str, Any]] = []
    for pid in ids:
        if pid == "pp_0":
            paraphrases.append({"paraphrase_id": pid})  # inherits base template
        else:
            paraphrases.append(
                {"paraphrase_id": pid, "templates": {"likert5": FAMILY_TEMPLATES[pid]}}
            )
    config["prompt"] = {
        "prompt_id": "p_test",
        "prompt_family_id": "pf_test",
        "templates": {"likert5": FAMILY_TEMPLATES["pp_0"]},
        "paraphrases": paraphrases,
    }
    config["repeats"] = {"n": n_repeats, "max_attempts_factor": 2, "seed_base": 1000}
    config["pilot"] = {"n_items": 2, "n_repeats": n_repeats}
    return config


def records_by_cell(records: List[ScoreRecord]) -> Dict[Any, List[ScoreRecord]]:
    """Group records by (env_id, item_id, system_id), each sorted by repeat_idx."""
    grouped: Dict[Any, List[ScoreRecord]] = {}
    for record in records:
        key = (record["env_id"], record["item_id"], record["system_id"])
        grouped.setdefault(key, []).append(record)
    for group in grouped.values():
        group.sort(key=lambda r: r["repeat_idx"])
    return grouped


def test_family_config_yields_one_env_id_across_all_paraphrases(
    config: Dict[str, Any], data_dir: str
) -> None:
    """A declared paraphrase family collapses to ONE env_id (ADR-011 D1 needs omega^2 in-env)."""
    # Given: a single scale/task/temperature grid with a 4-paraphrase family
    config = family_config(config)
    # When: the config is planned
    plan = runner.plan_run(config, pilot=False, data_dir=data_dir)
    # Then: every cell shares one env_id and carries the whole family in declared order
    env_ids = {cell.env_id for cell in plan.cells}
    assert len(env_ids) == 1
    assert all(len(cell.paraphrases) == 4 for cell in plan.cells)
    assert all(
        [p.paraphrase_id for p in cell.paraphrases] == list(FAMILY_PARAPHRASE_IDS)
        for cell in plan.cells
    )


def test_family_records_carry_per_paraphrase_id_and_a_distinct_payload_hash(
    config: Dict[str, Any], data_dir: str
) -> None:
    """Each record keeps its OWN paraphrase_id and its own rendered-prompt hash (ADR-009)."""
    # Given: a completed family run (8 repeats over 4 paraphrases)
    config = family_config(config, n_repeats=8)
    open_gate(config, data_dir)
    runner.run(config, pilot=True, data_dir=data_dir, client=MockJudgeClient())
    # When: records are grouped by cell
    cells = records_by_cell(all_records(config, data_dir))
    # Then: every cell shows all four paraphrases, each with exactly one payload hash,
    # and the four hashes are mutually distinct (each paraphrase renders a different prompt)
    assert cells
    for cell_records in cells.values():
        hashes: Dict[str, Set[str]] = {}
        for record in cell_records:
            hashes.setdefault(record["paraphrase_id"], set()).add(record["request_payload_hash"])
        assert set(hashes) == set(FAMILY_PARAPHRASE_IDS)
        assert all(len(h) == 1 for h in hashes.values())
        distinct = {next(iter(h)) for h in hashes.values()}
        assert len(distinct) == 4


def test_family_repeats_are_distributed_round_robin_deterministically(
    config: Dict[str, Any], data_dir: str
) -> None:
    """N repeats rotate across the family by repeat_idx: N=8, k=4 -> 2 each, in order."""
    # Given: a completed family run of 8 repeats over 4 paraphrases
    config = family_config(config, n_repeats=8)
    open_gate(config, data_dir)
    runner.run(config, pilot=True, data_dir=data_dir, client=MockJudgeClient())
    expected = [FAMILY_PARAPHRASE_IDS[idx % 4] for idx in range(8)]
    # When / Then: within every cell, paraphrase_id by repeat_idx is the round-robin rotation
    cells = records_by_cell(all_records(config, data_dir))
    assert cells
    for cell_records in cells.values():
        assert [record["repeat_idx"] for record in cell_records] == list(range(8))
        assert [record["paraphrase_id"] for record in cell_records] == expected
        counts = {pid: expected.count(pid) for pid in FAMILY_PARAPHRASE_IDS}
        assert set(counts.values()) == {2}
    # And: re-planning is byte-stable, so the rotation is reproducible
    assert (
        runner.plan_run(config, pilot=True, data_dir=data_dir).cells
        == runner.plan_run(config, pilot=True, data_dir=data_dir).cells
    )


def test_legacy_single_paraphrase_config_env_id_is_byte_identical(
    config: Dict[str, Any], data_dir: str
) -> None:
    """Regression guard: a config with no paraphrases list keeps its exact pre-family env_id."""
    # Given: the offline base config (single paraphrase_id, no paraphrases list)
    # When: it is planned
    plan = runner.plan_run(config, pilot=False, data_dir=data_dir)
    env_ids = sorted({cell.env_id for cell in plan.cells})
    # Then: the env_id equals the value computed before the family change landed
    assert env_ids == ["e_04d31761320f"]
    # And: the single paraphrase reproduces the legacy paraphrase_id and payload hash
    cell = plan.cells[0]
    assert len(cell.paraphrases) == 1
    assert cell.paraphrases[0].paraphrase_id == "pp_0"


def test_existing_yaml_configs_env_ids_are_unchanged(data_dir: str) -> None:
    """The shipped exp1_dense env_ids must not drift (offline; skipped on a cold cache)."""
    # Given: the real exp1_dense config, resolved from the pinned input cache only
    config = runner.load_config(os.path.join("configs", "exp1_dense.yaml"))
    try:
        plan = runner.plan_run(config, pilot=False, data_dir="data", allow_fetch=False)
    except FileNotFoundError:
        pytest.skip("exp1_dense input cache is cold; run estimate-cost once with network")
    # When / Then: the three per-scale env_ids equal the pre-family values
    assert sorted({cell.env_id for cell in plan.cells}) == [
        "e_200f3c74cbfa",
        "e_250190cb1b81",
        "e_7268666f4d91",
    ]


def test_decay_grouped_cells_finds_multiple_paraphrase_groups_per_cell(
    config: Dict[str, Any], data_dir: str
) -> None:
    """A family run gives decay.grouped_cells >1 paraphrase group per cell — omega^2 measurable."""
    # Given: a completed family run (8 repeats over 4 paraphrases -> 2 per group)
    config = family_config(config, n_repeats=8)
    open_gate(config, data_dir)
    runner.run(config, pilot=True, data_dir=data_dir, client=MockJudgeClient())
    records = all_records(config, data_dir)
    (family_env_id,) = {record["env_id"] for record in records}
    # When: the decay grouping runs over the single family env (group_field=paraphrase_id)
    cells = grouped_cells(records, family_env_id)
    # Then: every cell holds all four paraphrase groups, each with >= 2 repeats
    assert cells
    assert all(len(groups) == 4 for groups in cells)
    assert all(len(group) >= 2 for groups in cells for group in groups)
