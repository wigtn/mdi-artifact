"""Tests for the mdi CLI surface (PRD §5.1)."""

import os
from typing import Any, Dict, List, Tuple

import pytest
import yaml

from mdi import runner
from mdi.cli import main

HELP_ARGV: List[List[str]] = [
    ["--help"],
    ["estimate-cost", "--help"],
    ["run", "--help"],
    ["analyze", "--help"],
    ["census", "--help"],
    ["report", "--help"],
]

# analyze {variance,fip,decay,mdi-table} and report {figures,tables,all,cost} are wired to
# the analysis layer — their behaviour is covered by tests/test_stats_pipeline.py.
NOT_IMPLEMENTED_ARGV: List[Tuple[List[str], str]] = [
    (["analyze", "sources", "--exp", "exp3"], "FR-008"),
    (["census", "filter"], "FR-010"),
    (["census", "kappa"], "FR-010"),
    (["census", "merge"], "FR-010"),
    (["census", "bootstrap"], "FR-011"),
]


@pytest.fixture()
def config_path(config: Dict[str, Any], tmp_path: Any) -> str:
    """Write the offline test config to a YAML file the CLI can load."""
    path = os.path.join(str(tmp_path), "test.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=True)
    return path


@pytest.mark.parametrize("argv", HELP_ARGV, ids=[" ".join(a) for a in HELP_ARGV])
def test_help_exits_zero_for_each_subcommand(argv: List[str]) -> None:
    """--help must exit 0 for the top level and every subcommand."""
    # Given: a --help invocation
    # When: the CLI runs
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    # Then: argparse exits with status 0
    assert excinfo.value.code == 0


@pytest.mark.parametrize(
    ("argv", "fr"),
    NOT_IMPLEMENTED_ARGV,
    ids=[" ".join(a) for a, _ in NOT_IMPLEMENTED_ARGV],
)
def test_unimplemented_subcommand_exits_two_with_fr_message(
    argv: List[str], fr: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every remaining Phase 1 stub subcommand must exit 2 and name its FR on stderr."""
    # Given: a valid invocation of a not-yet-implemented subcommand
    # When: the CLI runs
    exit_code = main(argv)
    captured = capsys.readouterr()
    # Then: exit status 2 and a clear Phase 1 (FR-xxx) message on stderr
    assert exit_code == 2
    assert "not implemented" in captured.err
    assert "Phase 1" in captured.err
    assert fr in captured.err


def test_estimate_cost_prints_both_slices_and_writes_the_receipt(
    config_path: str, data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`mdi estimate-cost` reports the pilot and full slices and opens the run gate."""
    # Given: a config with an affordable cap
    # When: the estimator runs
    exit_code = main(["estimate-cost", "--config", config_path, "--data-dir", data_dir])
    out = capsys.readouterr().out
    # Then: it exits OK, prints both slices, and leaves a receipt behind
    assert exit_code == 0
    assert "[pilot slice]" in out
    assert "[full grid]" in out
    assert "VERDICT  : OK" in out
    config = runner.load_config(config_path)
    receipt = runner.estimate_receipt_path(runner.paths_for(data_dir), runner.config_hash(config))
    assert os.path.exists(receipt)


def test_estimate_cost_exits_nonzero_when_the_grid_exceeds_the_cap(
    config: Dict[str, Any], tmp_path: Any, data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A BLOCK verdict is a non-zero exit, so CI and shell pipelines stop."""
    # Given: an unaffordable cap
    config["budget"] = {"pilot_cap_usd": 1e-9, "cumulative_cap_usd": 1e-9}
    path = os.path.join(str(tmp_path), "blocked.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=True)
    # When: the estimator runs
    exit_code = main(["estimate-cost", "--config", path, "--data-dir", data_dir])
    captured = capsys.readouterr()
    # Then: non-zero exit and BLOCK on stderr
    assert exit_code == 1
    assert "BLOCK" in captured.err


def test_run_exits_nonzero_and_reports_the_gate_when_no_estimate_exists(
    config_path: str, data_dir: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`mdi run` without a prior estimate fails loudly instead of calling a judge API."""
    # Given: no gate receipt
    # When: a pilot run is attempted
    exit_code = main(["run", "--config", config_path, "--pilot", "--data-dir", data_dir])
    captured = capsys.readouterr()
    # Then: non-zero exit naming the gate
    assert exit_code == 1
    assert "EstimateRequired" in captured.err
    assert "estimate-cost" in captured.err
