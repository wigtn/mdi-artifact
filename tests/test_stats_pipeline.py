"""Tests for the analyze/report CLI wiring and the derived-artifact pipeline (FR-014)."""

import json
import os
from pathlib import Path
from typing import List

import pytest

from mdi import store
from mdi.cli import main
from mdi.stats import pipeline
from mdi.stats.synthetic import synthetic_records

ENV = "e_demo0001"
EXP = "exp2"


@pytest.fixture()
def data_dir(tmp_path: Path) -> str:
    """A scratch data tree holding one synthetic raw shard (never the repo's data/)."""
    raw_dir = tmp_path / "raw" / EXP
    raw_dir.mkdir(parents=True)
    records = synthetic_records(
        system_means={"s_A": 0.0, "s_B": 0.3, "s_C": 0.9, "s_D": 1.8},
        n_items=5,
        n_repeats=12,
        sigma=1.0,
        seed=51,
        env_id=ENV,
        group_offsets=(-0.3, 0.3),
    )
    with store.ShardWriter(str(raw_dir / f"{ENV}.jsonl")) as writer:
        for record in records:
            writer.append(record)
    return str(tmp_path)


def _analyze(data_dir: str, what: str, *extra: str) -> int:
    """Run one `mdi analyze` invocation against the scratch data tree."""
    return main(["analyze", what, "--exp", EXP, "--data-dir", data_dir, *extra])


def test_analyze_writes_a_derived_payload_with_full_provenance(data_dir: str) -> None:
    """Every derived payload must be traceable to run ids and an input digest (AGENTS.md §3.6)."""
    # Given: a scratch store
    # When: the variance analysis runs
    exit_code = _analyze(data_dir, "variance")
    payload = json.loads(
        Path(data_dir, "derived", EXP, "variance.json").read_text(encoding="utf-8")
    )
    # Then: it exits 0 and records schema, runs, digest and parameters
    assert exit_code == 0
    meta = payload["meta"]
    assert meta["analysis"] == "variance"
    assert meta["exp"] == EXP
    assert meta["schema_version"] == store.SCHEMA_VERSION
    assert meta["run_ids"] == ["r_synthetic"]
    assert meta["env_ids"] == [ENV]
    assert meta["input_digest"].startswith("sha256:")
    assert meta["params"]["seed"] == 20260731
    assert payload["envs"][0]["sigma"] == pytest.approx(1.0, rel=0.1)


def test_analyze_defaults_hold_out_repeat_zero_for_screening(data_dir: str) -> None:
    """The ADR-007 split is enforced by default, not left to the caller."""
    # Given: a store with repeats 0..11
    # When: the FIP analysis runs with default split arguments
    exit_code = _analyze(data_dir, "fip", "--draws", "60", "--resamples", "50")
    payload = json.loads(Path(data_dir, "derived", EXP, "fip.json").read_text(encoding="utf-8"))
    # Then: repeat 0 is screening-only and never enters estimation
    assert exit_code == 0
    params = payload["meta"]["params"]
    assert params["screening_repeats"] == [0]
    assert params["estimation_repeats"] == list(range(1, 12))
    curve = payload["curves"][0]
    assert curve["meta"]["screening_repeats"] == [0]


def test_analyze_fails_cleanly_when_the_store_is_missing(tmp_path: Path) -> None:
    """A missing store exits 1 with an actionable message, never a traceback."""
    # Given: an empty data tree
    # When: an analysis is requested
    exit_code = main(["analyze", "variance", "--exp", "nope", "--data-dir", str(tmp_path)])
    # Then: exit status 1
    assert exit_code == 1


def test_analyze_rejects_an_overlapping_repeat_split(data_dir: str) -> None:
    """The ADR-007 guard surfaces through the CLI as a clean failure."""
    # Given: a split that reuses the screening repeat for estimation
    # When: the FIP analysis runs
    exit_code = _analyze(
        data_dir, "fip", "--screening-repeats", "0-2", "--estimation-repeats", "2-11"
    )
    # Then: exit status 1 (OverlappingRepeatsError is a ValueError)
    assert exit_code == 1


def test_analyze_sources_remains_a_declared_stub(data_dir: str) -> None:
    """FR-008 is out of this slice's scope and must still announce itself."""
    # Given/When: the variance-sources analysis is requested
    exit_code = _analyze(data_dir, "sources")
    # Then: the Phase 1 stub contract (exit 2) is preserved
    assert exit_code == 2


def test_report_all_is_byte_identical_across_two_runs(data_dir: str, tmp_path: Path) -> None:
    """`mdi report all` twice must produce byte-identical artifacts (AGENTS.md §3.3)."""
    # Given: all four analyses written to the derived tree
    assert _analyze(data_dir, "variance") == 0
    assert (
        _analyze(data_dir, "fip", "--n-repeats", "1,5", "--draws", "60", "--resamples", "50") == 0
    )
    assert _analyze(data_dir, "decay", "--draws", "150") == 0
    assert (
        _analyze(data_dir, "mdi-table", "--sweep", "1,5", "--draws", "60", "--resamples", "50") == 0
    )
    first_out = tmp_path / "paper_a"
    second_out = tmp_path / "paper_b"
    # When: the report is regenerated twice
    assert main(["report", "all", "--data-dir", data_dir, "--out", str(first_out)]) == 0
    assert main(["report", "all", "--data-dir", data_dir, "--out", str(second_out)]) == 0
    # Then: the two trees match file-for-file and byte-for-byte
    first_files = sorted(str(p.relative_to(first_out)) for p in first_out.rglob("*") if p.is_file())
    second_files = sorted(
        str(p.relative_to(second_out)) for p in second_out.rglob("*") if p.is_file()
    )
    assert first_files == second_files
    assert first_files, "the report produced no artifacts"
    for name in first_files:
        assert (first_out / name).read_bytes() == (second_out / name).read_bytes(), name


def test_report_writes_the_mdi_and_conversion_tables(data_dir: str, tmp_path: Path) -> None:
    """Tables carry both MDI paths (null-PI primary, %p first) with their provenance columns."""
    # Given: the FIP and MDI analyses
    _analyze(data_dir, "fip", "--draws", "60", "--resamples", "50")
    _analyze(
        data_dir,
        "mdi-table",
        "--sweep",
        "1,5",
        "--draws",
        "60",
        "--null-draws",
        "120",
        "--resamples",
        "50",
    )
    out = tmp_path / "paper"
    # When: tables are regenerated
    exit_code = main(["report", "tables", "--data-dir", data_dir, "--out", str(out)])
    mdi_csv = (out / "tables" / "mdi_table.csv").read_text(encoding="utf-8")
    conversion_csv = (out / "tables" / "fip_conversion.csv").read_text(encoding="utf-8")
    # Then: both tables exist with %p-first headers and LF endings
    assert exit_code == 0
    assert mdi_csv.splitlines()[0] == (
        "exp,task,scale,env_id,n_repeats,alpha,mdi_null_pp,mdi_null_smoothed_pp,"
        "mdi_null_ci_lo_pp,mdi_null_ci_hi_pp,power,mdi_power_pp,ratio_power,"
        "pool_inflation,mdi_null_corrected_pp,mdi_fip_pp,ratio_fip_over_null,abs_diff_pp,"
        "mdi_null,mdi_fip,attained_fip,sigma,readoff,n_draws_null,n_draws_fip"
    )
    assert "\r" not in mdi_csv
    assert conversion_csv.splitlines()[0] == (
        "exp,env_id,task,scale,n_repeats,delta_lo_pp,delta_hi_pp,delta_pp,reversal_pct,"
        "ci_lo_pct,ci_hi_pct,n_draws,delta_lo,delta_hi,delta,statement"
    )
    assert "reverses" in conversion_csv


def test_report_tables_renders_the_sla_margin_tex(data_dir: str, tmp_path: Path) -> None:
    """The SLA margin table is the alpha=0.10 |delta| 0.90-quantile, rendered with provenance."""
    # Given: variance (judge map) + mdi-table (null entries) payloads
    assert _analyze(data_dir, "variance") == 0
    assert (
        _analyze(data_dir, "mdi-table", "--sweep", "1,5", "--draws", "60", "--resamples", "50") == 0
    )
    out = tmp_path / "paper"
    # When: the renderer runs over a published set holding this fixture (ADR-032;
    # the CLI path uses the production set, which the two tests below cover)
    path = pipeline.report_sla_margin_tex(
        os.path.join(data_dir, "derived"), str(out / "tables"), published=frozenset({EXP})
    )
    assert path is not None
    text = Path(path).read_text(encoding="utf-8")
    # Then: the tex table exists, carries provenance, and its N=1 cell is the
    # recorded one-sided margin (abs_delta_quantile_pp of the alpha=0.10 entry)
    # tabular* + \extracolsep: the generated tables span the text width
    # (first-author layout note 2026-08-06), so the environment is starred.
    assert "\\begin{tabular*}{\\linewidth}" in text and "\\bottomrule" in text
    assert "% Provenance:" in text and "run_ids [r_synthetic]" in text
    payload = json.loads(
        Path(data_dir, "derived", EXP, "mdi_table.json").read_text(encoding="utf-8")
    )
    entry = next(
        e for e in payload["null_entries"] if abs(e["alpha"] - 0.10) < 1e-12 and e["n_repeats"] == 1
    )
    assert f"{entry['abs_delta_quantile_pp']:.2f}" in text


def test_report_cost_tex_prices_tiers_from_the_ledger(data_dir: str, tmp_path: Path) -> None:
    """Cost cells are ledger spend / calls x N x 100; unmapped rows are named, never dropped."""
    # Given: a variance payload mapping EXP to its judge, and a two-row ledger
    assert _analyze(data_dir, "variance") == 0
    ledger_rows = [
        {"experiment": EXP, "calls": 100, "cost_usd": 1.0, "run_id": "r_synthetic"},
        {"experiment": "spike_unmapped", "calls": 10, "cost_usd": 5.0, "run_id": "r_x"},
    ]
    Path(data_dir, "ledger.jsonl").write_text(
        "\n".join(json.dumps(row) for row in ledger_rows) + "\n", encoding="utf-8"
    )
    out = tmp_path / "paper"
    # When: the renderer runs over a published set holding this fixture (ADR-032)
    path = pipeline.report_cost_tex(
        os.path.join(data_dir, "derived"),
        str(out / "tables"),
        os.path.join(data_dir, "ledger.jsonl"),
        published=frozenset({EXP}),
    )
    assert path is not None
    text = Path(path).read_text(encoding="utf-8")
    # Then: per-call $0.01 -> N=3 over 100 cells = $3.000, and the unmapped row is named
    assert "3.000" in text
    assert "skipped (no derived tier mapping): spike_unmapped" in text
    assert "sha256:" in text


def test_rendered_tables_ignore_an_experiment_outside_the_published_set() -> None:
    """A validation run must not reach a rendered table just by existing (ADR-032)."""
    # Given: the production default
    published = pipeline.PUBLISHED_EXPERIMENTS
    # Then: the measurement grids are in it and the ADR-029 §e ladder is not —
    # the ladder shares exp1_dense's env_id, so inclusion would displace a
    # published row rather than add one
    assert {"exp1_dense", "exp1_alpaca", "exp1_anchor", "exp1_anchor_scales"} == set(published)
    assert "exp1_ladder" not in published
    assert not any(name.startswith("spike") for name in published)


def test_unpublished_experiments_render_nothing_rather_than_a_partial_table(
    data_dir: str, tmp_path: Path
) -> None:
    """An empty published set yields no table at all, never a half-filled one."""
    # Given: a derived tree whose only experiment is outside the published set
    assert _analyze(data_dir, "variance") == 0
    assert (
        _analyze(data_dir, "mdi-table", "--sweep", "1,5", "--draws", "60", "--resamples", "50") == 0
    )
    # When: the renderers run with that experiment excluded
    derived = os.path.join(data_dir, "derived")
    sla = pipeline.report_sla_margin_tex(
        derived, str(tmp_path / "t"), published=frozenset({"exp1_dense"})
    )
    cost = pipeline.report_cost_tex(
        derived, str(tmp_path / "t"), published=frozenset({"exp1_dense"})
    )
    # Then: nothing is rendered, and no file is left behind to be mistaken for one
    assert sla is None
    assert cost is None
    assert not (tmp_path / "t" / "sla_margin.tex").exists()


def test_report_cost_tex_returns_none_without_a_ledger(data_dir: str, tmp_path: Path) -> None:
    """No ledger -> no cost table and no crash (the artifact is measured spend only)."""
    # Given: a derived tree but no ledger.jsonl
    assert _analyze(data_dir, "variance") == 0
    # When: the renderer is called directly
    result = pipeline.report_cost_tex(os.path.join(data_dir, "derived"), str(tmp_path / "tables"))
    # Then: it declines to invent numbers
    assert result is None


def test_mdi_table_payload_carries_both_paths_and_the_consistency_block(data_dir: str) -> None:
    """ADR-015: primary null-PI entries, validation entries and their consistency check."""
    # Given/When: the mdi-table analysis runs
    exit_code = _analyze(
        data_dir,
        "mdi-table",
        "--sweep",
        "1,5",
        "--draws",
        "60",
        "--null-draws",
        "120",
        "--resamples",
        "50",
    )
    payload = json.loads(
        Path(data_dir, "derived", EXP, "mdi_table.json").read_text(encoding="utf-8")
    )
    # Then: both paths are present and the consistency block compares them per key
    assert exit_code == 0
    assert payload["primary"] == "null_pi_half_width"
    assert payload["validation"] == "fip_crossing"
    assert payload["null_entries"], "primary path missing"
    assert payload["entries"], "validation path missing"
    consistency = payload["consistency"]
    assert consistency
    for block in consistency:
        assert set(block) >= {"mdi_null", "mdi_fip_crossing", "ratio", "abs_diff"}
        if block["mdi_null"] is not None and block["mdi_fip_crossing"] is not None:
            expected = abs(block["mdi_null"] - block["mdi_fip_crossing"])
            assert block["abs_diff"] == pytest.approx(expected)


def test_report_reports_nothing_to_render_on_an_empty_derived_tree(tmp_path: Path) -> None:
    """An empty derived tree exits 1 and points at `mdi analyze`."""
    # Given: no derived payloads
    # When: a report is requested
    exit_code = main(
        ["report", "figures", "--data-dir", str(tmp_path), "--out", str(tmp_path / "paper")]
    )
    # Then: a clean failure
    assert exit_code == 1


def test_report_cost_on_an_empty_tree_reports_nothing_to_render(tmp_path: Path) -> None:
    """FR-012 is implemented: with no derived tree and no ledger there is nothing to price."""
    # Given/When: the cost report is requested against an empty data dir
    exit_code = main(["report", "cost", "--data-dir", str(tmp_path)])
    # Then: the CLI reports "nothing to render" (exit 1), never invented numbers
    assert exit_code == 1


def test_analysis_json_is_written_with_sorted_keys_and_a_trailing_newline(
    data_dir: str, tmp_path: Path
) -> None:
    """Derived JSON must be diff-stable."""
    # Given: a written payload
    _analyze(data_dir, "variance")
    text = Path(data_dir, "derived", EXP, "variance.json").read_text(encoding="utf-8")
    # When/Then: keys are sorted, ASCII-escaped, LF-terminated
    assert text.endswith("}\n")
    assert "\r" not in text
    reparsed = json.dumps(json.loads(text), sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    assert text == reparsed


@pytest.mark.parametrize(
    ("spec", "expected"),
    [("0", [0]), ("1,3,5", [1, 3, 5]), ("0-3", [0, 1, 2, 3]), ("", []), (" 2 , 2 ", [2])],
)
def test_repeat_spec_parsing_accepts_lists_and_ranges(spec: str, expected: List[int]) -> None:
    """Repeat specs accept single values, comma lists and inclusive ranges."""
    # Given/When/Then
    assert pipeline.parse_int_spec(spec) == expected


def test_repeat_spec_parsing_rejects_a_reversed_range() -> None:
    """A reversed range is a typo, not an empty set."""
    # Given/When/Then
    with pytest.raises(pipeline.AnalysisError, match="invalid range"):
        pipeline.parse_int_spec("5-1")


def test_shard_paths_ignores_quarantine_sidecars(tmp_path: Path) -> None:
    """Quarantined lines are provenance, not analysis input (PRD §5.2)."""
    # Given: a shard plus its quarantine sidecar
    (tmp_path / "e_x.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "e_x.jsonl.quarantine.jsonl").write_text("", encoding="utf-8")
    # When: shards are enumerated
    paths = pipeline.shard_paths(str(tmp_path))
    # Then: only the shard is loaded
    assert [os.path.basename(path) for path in paths] == ["e_x.jsonl"]
