"""mdi command-line entry point (PRD §5.1).

Subcommands: estimate-cost / run / analyze / census / report. All experiment
execution goes through this CLI — never ad-hoc scripts against judge APIs
(AGENTS.md §3.2). Unimplemented subcommands exit with status 2 and a Phase 1
FR pointer on stderr; ``--help`` always exits 0.

``estimate-cost`` and ``run`` are wired to the Phase 1 runner. The gate order
is mechanical, not advisory: ``estimate-cost`` writes a receipt keyed by the
canonical config hash, ``run --pilot`` needs that receipt, and a full ``run``
additionally needs the pilot-completion flag.

``analyze`` and ``report`` are wired to the analysis layer (:mod:`mdi.stats`):
``analyze`` reads the raw store and writes a derived payload per experiment
under ``data/derived/<exp>/``; ``report`` renders ``paper/figures/`` and
``paper/tables/`` from those payloads and from nothing else, deterministically
(AGENTS.md §3.3 — two runs must be byte-identical).
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

import dotenv

from mdi import cost, runner, store
from mdi.stats import (
    bootstrap,
    decay,
    figures,
    mdi_table,
    null_mdi,
    pipeline,
    promotion,
    variance,
)

DEFAULT_PAPER_DIR: str = "paper"

_ANALYZE_FR: Dict[str, str] = {
    "variance": "FR-005",
    "fip": "FR-006",
    "decay": "FR-007",
    "mdi-table": "FR-009",
    "sources": "FR-008",
    "promotions": "FR-011",
    # ADR-029: external-oracle validation. No FR of its own — it validates
    # FR-009's threshold rather than adding a requirement.
    "oracle": "ADR-029",
}

_CENSUS_FR: Dict[str, str] = {
    "filter": "FR-010",
    "kappa": "FR-010",
    "merge": "FR-010",
    "bootstrap": "FR-011",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree matching the PRD §5.1 CLI specification."""
    parser = argparse.ArgumentParser(
        prog="mdi",
        description="MDI estimation framework — grid judging, analysis, census, reporting.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_est = sub.add_parser(
        "estimate-cost",
        help="Dry-run cost estimate for a config grid; required before any run (FR-004)",
    )
    p_est.add_argument("--config", required=True, metavar="PATH", help="Experiment YAML config")
    p_est.add_argument(
        "--data-dir",
        default=runner.DEFAULT_DATA_DIR,
        metavar="DIR",
        help="Data tree root (default: data)",
    )

    p_run = sub.add_parser("run", help="Repeated judging run (FR-002)")
    p_run.add_argument("--config", required=True, metavar="PATH", help="Experiment YAML config")
    p_run.add_argument("--resume", metavar="RUN_ID", default=None, help="Resume an interrupted run")
    p_run.add_argument("--pilot", action="store_true", help="Run only the config's pilot slice")
    p_run.add_argument(
        "--data-dir",
        default=runner.DEFAULT_DATA_DIR,
        metavar="DIR",
        help="Data tree root (default: data)",
    )

    p_analyze = sub.add_parser(
        "analyze", help="Analyses over the raw store (FR-005..FR-009, FR-011)"
    )
    p_analyze.add_argument(
        "what",
        choices=["variance", "fip", "decay", "mdi-table", "sources", "promotions", "oracle"],
        help="Which analysis to run",
    )
    p_analyze.add_argument(
        "--facets",
        default=",".join(store.SUMMEVAL_FACETS),
        metavar="LIST",
        help=(
            "oracle: annotation facets averaged with equal weight "
            f"(default: {','.join(store.SUMMEVAL_FACETS)})"
        ),
    )
    p_analyze.add_argument(
        "--alpha-oracle",
        type=float,
        default=mdi_table.DEFAULT_ALPHA,
        metavar="FLOAT",
        help=(
            "oracle: which MDI column the claim rates are measured against "
            f"(default: {mdi_table.DEFAULT_ALPHA})"
        ),
    )
    p_analyze.add_argument("--exp", required=True, metavar="EXP_ID", help="Experiment id")
    p_analyze.add_argument(
        "--source",
        default="regimes",
        choices=["regimes"],
        help="promotions: external dataset for the Exp 4 case study (default: regimes)",
    )
    p_analyze.add_argument(
        "--promotion-alpha",
        type=float,
        default=promotion.DEFAULT_ALPHA,
        metavar="FLOAT",
        help=(
            "promotions: noise-band significance for the stopping counterfactual "
            f"(default: {promotion.DEFAULT_ALPHA})"
        ),
    )
    p_analyze.add_argument(
        "--data-dir",
        default=runner.DEFAULT_DATA_DIR,
        metavar="DIR",
        help="Data tree root (default: data)",
    )
    p_analyze.add_argument(
        "--store",
        default=None,
        metavar="PATH",
        help="Raw store shard file or directory (default: <data-dir>/raw/<exp>)",
    )
    p_analyze.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="Derived output directory (default: <data-dir>/derived/<exp>)",
    )
    p_analyze.add_argument(
        "--env",
        action="append",
        default=None,
        metavar="ENV_ID",
        help="Restrict to this env_id (repeatable; default: every env in the store)",
    )
    p_analyze.add_argument(
        "--seed",
        type=int,
        default=bootstrap.DEFAULT_SEED,
        metavar="INT",
        help=f"Analysis seed (default: {bootstrap.DEFAULT_SEED})",
    )
    p_analyze.add_argument(
        "--screening-repeats",
        default=pipeline.DEFAULT_SCREENING_REPEATS,
        metavar="SPEC",
        help="ADR-007 held-out screening repeat indices, e.g. '0' or '0-1' (default: 0)",
    )
    p_analyze.add_argument(
        "--estimation-repeats",
        default=None,
        metavar="SPEC",
        help="Repeat indices used for FIP estimation (default: every repeat not screened)",
    )
    p_analyze.add_argument(
        "--n-repeats",
        default="1",
        metavar="SPEC",
        help="fip: evaluation budgets N to estimate, e.g. '1,5' (default: 1)",
    )
    p_analyze.add_argument(
        "--sweep",
        default="1,3,5,8,10,20",
        metavar="SPEC",
        help="decay/mdi-table: repeat sweep (default: 1,3,5,8,10,20 — ADR-013 adds N=8)",
    )
    p_analyze.add_argument(
        "--alphas",
        default="0.10,0.05,0.01",
        metavar="SPEC",
        help="mdi-table: error levels to tabulate (default: 0.10,0.05,0.01)",
    )
    p_analyze.add_argument(
        "--draws",
        type=int,
        default=None,
        metavar="INT",
        help="Draws per pair (fip/mdi-table) or per cell (decay)",
    )
    p_analyze.add_argument(
        "--null-draws",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "mdi-table: null-distribution draws per system for the primary "
            "null-PI MDI path (ADR-015a; default: "
            f"{null_mdi.DEFAULT_DRAWS_PER_SYSTEM})"
        ),
    )
    p_analyze.add_argument(
        "--resamples",
        type=int,
        default=bootstrap.DEFAULT_N_RESAMPLES,
        metavar="INT",
        help=f"Bootstrap resamples (default: {bootstrap.DEFAULT_N_RESAMPLES})",
    )
    p_analyze.add_argument(
        "--ci-level",
        type=float,
        default=bootstrap.DEFAULT_CI_LEVEL,
        metavar="FLOAT",
        help=f"Bootstrap CI level (default: {bootstrap.DEFAULT_CI_LEVEL})",
    )
    p_analyze.add_argument(
        "--group-field",
        default=decay.DEFAULT_GROUP_FIELD,
        metavar="FIELD",
        help=f"decay: procedural grouping field (default: {decay.DEFAULT_GROUP_FIELD})",
    )
    p_analyze.add_argument(
        "--alpha-level",
        default=variance.DEFAULT_ALPHA_LEVEL,
        choices=[variance.INTERVAL, variance.NOMINAL, variance.ORDINAL],
        help=f"variance: Krippendorff difference metric (default: {variance.DEFAULT_ALPHA_LEVEL})",
    )
    p_analyze.add_argument(
        "--readoff",
        default=mdi_table.DEFAULT_READOFF,
        choices=[mdi_table.INTERPOLATE, mdi_table.BIN_UPPER],
        help=f"mdi-table: FIP read-off rule (default: {mdi_table.DEFAULT_READOFF})",
    )
    p_analyze.add_argument(
        "--targets-pp",
        default=None,
        metavar="SPEC",
        help=(
            "mdi-table: target improvements (%%p) for the reverse 'how many "
            "repeats?' table, e.g. '0.5,1,2' (default: "
            f"{','.join(str(t) for t in mdi_table.DEFAULT_TARGETS_PP)})"
        ),
    )

    p_census = sub.add_parser("census", help="Exp 4 census pipeline (FR-010/FR-011)")
    p_census.add_argument(
        "step",
        choices=["filter", "kappa", "merge", "bootstrap"],
        help="Census pipeline step",
    )

    p_report = sub.add_parser(
        "report", help="Deterministic regeneration of paper artifacts (FR-014)"
    )
    p_report.add_argument(
        "target",
        choices=["all", "figures", "tables", "cost"],
        help="Which artifacts to regenerate",
    )
    p_report.add_argument(
        "--data-dir",
        default=runner.DEFAULT_DATA_DIR,
        metavar="DIR",
        help="Data tree root (default: data)",
    )
    p_report.add_argument(
        "--derived",
        default=None,
        metavar="DIR",
        help="Derived analysis tree to render from (default: <data-dir>/derived)",
    )
    p_report.add_argument(
        "--out",
        default=DEFAULT_PAPER_DIR,
        metavar="DIR",
        help=f"Paper artifact root (default: {DEFAULT_PAPER_DIR})",
    )
    p_report.add_argument(
        "--alpha",
        type=float,
        default=mdi_table.DEFAULT_ALPHA,
        metavar="FLOAT",
        help=f"Error level marked on the FIP figures (default: {mdi_table.DEFAULT_ALPHA})",
    )
    p_report.add_argument(
        "--formats",
        default=",".join(figures.DEFAULT_FORMATS),
        metavar="LIST",
        help=f"Figure formats (default: {','.join(figures.DEFAULT_FORMATS)})",
    )

    return parser


def _not_implemented(command: str, fr: str) -> int:
    """Print the Phase 1 stub notice to stderr and return exit status 2."""
    print(f"mdi {command}: not implemented — Phase 1 ({fr})", file=sys.stderr)
    return 2


def _print_slice(label: str, report: Dict[str, Any]) -> None:
    """Print one slice of an estimate report (pilot or full)."""
    expected = report["expected"]
    worst = report["worst_case"]
    verdict = report["verdict"]
    print(f"\n[{label}]  config_hash={report['config_hash']}  inputs={report['inputs_digest']}")
    print(
        f"  cells={report['cells']} (items={report['items']}, "
        f"systems={len(report['systems'])})  N={report['n_repeats']}  "
        f"attempt cap={report['max_attempts']}"
    )
    print(
        f"  expected : {expected['calls']:>7} calls  "
        f"{expected['in_tokens']:>9} in / {expected['out_tokens']:>7} out tok  "
        f"${expected['cost_usd']:.4f}"
    )
    print(
        f"  worst 2N : {worst['calls']:>7} calls  "
        f"{worst['in_tokens']:>9} in / {worst['out_tokens']:>7} out tok  "
        f"${worst['cost_usd']:.4f}"
    )
    for entry in worst["by_model"]:
        print(f"    - {entry['model']}: ${entry['cost_usd']:.4f} over {entry['calls']} calls")
    print(
        f"  budget   : ledger ${verdict['spent_usd']:.4f} + projected "
        f"${verdict['projected_usd']:.4f} vs cap ${verdict['slice_cap_usd']:.2f} "
        f"(cumulative ${verdict['cumulative_cap_usd']:.2f})"
    )
    print(f"  VERDICT  : {verdict['verdict']} — {verdict['reason']}")


def _estimate_cost(config_path: str, data_dir: str) -> int:
    """Run the dry-run estimator for both slices and write the gate receipt."""
    config = runner.load_config(config_path)
    paths = runner.paths_for(data_dir)
    pilot_report = runner.estimate(config, pilot=True, data_dir=data_dir)
    full_report = runner.estimate(config, pilot=False, data_dir=data_dir)

    print(f"mdi estimate-cost — {config_path} (experiment: {full_report['experiment']})")
    _print_slice("pilot slice", pilot_report)
    _print_slice("full grid", full_report)

    receipt = {
        "config_path": config_path,
        "config_hash": full_report["config_hash"],
        "pilot": pilot_report,
        "full": full_report,
    }
    path = runner.write_estimate_receipt(paths, full_report["config_hash"], receipt)
    print(f"\ngate receipt: {path}")

    blocked = [
        label
        for label, report in (("pilot", pilot_report), ("full", full_report))
        if report["verdict"]["verdict"] == cost.VERDICT_BLOCK
    ]
    if blocked:
        print(f"BLOCK: {', '.join(blocked)} slice(s) exceed the cap", file=sys.stderr)
        return 1
    return 0


def _run(config_path: str, data_dir: str, resume: Optional[str], pilot: bool) -> int:
    """Execute a judging run behind the cost/pilot gates."""
    config = runner.load_config(config_path)
    try:
        run_id = runner.run(config, resume=resume, pilot=pilot, data_dir=data_dir)
    except (runner.BudgetExceeded, runner.PilotRequired, runner.ProviderError) as exc:
        print(f"mdi run: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"run_id: {run_id}")
    print(f"ledger: {runner.paths_for(data_dir).ledger_path}")
    return 0


def _print_promotions(report: promotion.PromotionReport, path: str) -> None:
    """Print the per-promotion table and per-seed stopping counterfactuals."""
    print("mdi analyze promotions — Exp 4 Regimes case study (ADR-017 / FR-011)")
    print(f"  source  : {store.REGIMES_REPO} @ {store.REGIMES_REF[:12]} ({store.REGIMES_LICENSE})")
    print(f"  alpha   : {report['alpha']}")
    print(f"  digest  : {report['input_digest']}")
    print(f"  {report['n_promotions']} promotions over {report['n_seeds']} seeds:")
    print(
        "    seed  #  name                          b   c   delta  mcnemar_p  reversal  band  halt"
    )
    for row in report["rows"]:
        print(
            f"    {row['seed']:>4}  {row['promo_num']}  {row['name'][:28]:28}  "
            f"{row['n_recovered']:>2}  {row['n_introduced']:>2}  {row['confirm_delta']:+.2f}  "
            f"{row['mcnemar_p']:>8.4f}  {row['reversal_prob']:>7.4f}  "
            f"{'yes' if row['clears_noise_band'] else 'no':>4}  "
            f"{'<= HALT' if row['would_halt_here'] else ''}"
        )
    print("  stopping counterfactual (our rule vs the loop's accept-if-better gate):")
    for result in report["stopping"]:
        if result["halt_promo_num"] is None:
            print(
                f"    seed {result['seed']}: no promotion clears the band — "
                "underpowered, loop should not have promoted"
            )
            continue
        print(
            f"    seed {result['seed']}: halt at promo #{result['halt_promo_num']} "
            f"({result['halt_n_recovered']} vs {result['halt_n_introduced']}), peak "
            f"{result['peak_confirm_delta']:+.2f} vs final {result['final_confirm_delta']:+.2f}, "
            f"{result['n_promotions_after_halt']} noise promotion(s) after"
        )
    print(f"  wrote   : {path}")


def _analyze_promotions(args: argparse.Namespace) -> int:
    """Exp 4 promotion-reversal case study over the pinned Regimes data (ADR-017 / FR-011)."""
    if args.source != "regimes":
        print(f"mdi analyze promotions: unknown --source {args.source!r}", file=sys.stderr)
        return 1
    inputs_dir = os.path.join(args.data_dir, "inputs")
    out_dir = args.out or pipeline.default_derived_dir(args.data_dir, args.exp)
    try:
        by_seed = store.load_regimes_promotions(inputs_dir, allow_fetch=True)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"mdi analyze promotions: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    report = promotion.promotion_report(by_seed, alpha=args.promotion_alpha)
    payload: Dict[str, Any] = dict(report)
    payload["meta"] = {
        "analysis": "promotions",
        "exp": args.exp,
        "schema_version": store.SCHEMA_VERSION,
        "source": args.source,
        "source_repo": store.REGIMES_REPO,
        "source_ref": store.REGIMES_REF,
        "license": store.REGIMES_LICENSE,
        "seeds": sorted(by_seed),
        "input_digest": report["input_digest"],
        "params": {"alpha": args.promotion_alpha},
    }
    path = pipeline.write_json(os.path.join(out_dir, "promotions.json"), payload)
    _print_promotions(report, path)
    return 0


def _analyze_oracle(args: argparse.Namespace) -> int:
    """External-oracle validation over the pinned SummEval annotations (ADR-029).

    Reads two things it does not compute: the annotation snapshot under
    ``data/inputs/`` (never fetched, never written) and this experiment's already
    derived MDI table, whose null entries supply the thresholds the claim rates
    are measured against. Running ``mdi analyze mdi-table`` first is therefore a
    precondition, and a missing table is reported as such rather than silently
    skipped.
    """
    inputs_dir = os.path.join(args.data_dir, "inputs")
    out_dir = args.out or pipeline.default_derived_dir(args.data_dir, args.exp)
    store_path = args.store or pipeline.default_store_path(args.data_dir, args.exp)
    facets = [chunk.strip() for chunk in args.facets.split(",") if chunk.strip()]
    table_path = os.path.join(out_dir, pipeline.ANALYSIS_FILES["mdi-table"])
    if not os.path.exists(table_path):
        print(
            f"mdi analyze oracle: no MDI table at {table_path} — "
            f"run `mdi analyze mdi-table --exp {args.exp}` first",
            file=sys.stderr,
        )
        return 1
    try:
        payload = pipeline.oracle_analysis(
            args.exp,
            store_path=store_path,
            out_dir=out_dir,
            inputs_dir=inputs_dir,
            facets=facets,
            env_ids=args.env,
            seed=args.seed,
            screening_spec=args.screening_repeats,
            estimation_spec=args.estimation_repeats,
            alpha=args.alpha_oracle,
            draws=args.draws,
            n_resamples=args.resamples,
            ci_level=args.ci_level,
        )
    except (pipeline.AnalysisError, FileNotFoundError, ValueError, OSError) as exc:
        print(f"mdi analyze oracle: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    path = pipeline.write_json(os.path.join(out_dir, pipeline.ANALYSIS_FILES["oracle"]), payload)
    print(f"mdi analyze oracle — exp {args.exp} ({_ANALYZE_FR['oracle']})")
    print(f"  facets  : {', '.join(sorted(facets))}")
    for report in payload["reports"]:
        concordance = report["rank_concordance"]
        agreement = "n/a" if concordance is None else f"{concordance:.2f}"
        print(
            f"  {report['env_id']} ({report['scale']}): "
            f"oracle spread {report['oracle_spread_pp']:.2f} %p vs "
            f"judge {report['judge_spread_pp']:.2f} %p, "
            f"{report['n_separated_pairs']}/{len(report['gaps'])} pairs separated externally, "
            f"rank agreement {agreement}"
        )
    print(f"  wrote   : {path}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    """Run one analysis and write its derived payload (FR-005..FR-009, FR-011)."""
    if args.what == "promotions":
        return _analyze_promotions(args)
    if args.what == "oracle":
        return _analyze_oracle(args)
    if args.what == "sources":
        return _not_implemented(f"analyze {args.what}", _ANALYZE_FR[args.what])
    store_path = args.store or pipeline.default_store_path(args.data_dir, args.exp)
    out_dir = args.out or pipeline.default_derived_dir(args.data_dir, args.exp)
    try:
        path, payload = pipeline.run_analysis(
            args.what,
            args.exp,
            store_path=store_path,
            out_dir=out_dir,
            env_ids=args.env,
            seed=args.seed,
            screening_spec=args.screening_repeats,
            estimation_spec=args.estimation_repeats,
            n_repeats_spec=args.n_repeats,
            sweep_spec=args.sweep,
            alphas_spec=args.alphas,
            draws=args.draws,
            n_resamples=args.resamples,
            ci_level=args.ci_level,
            group_field=args.group_field,
            alpha_level=args.alpha_level,
            readoff=args.readoff,
            null_draws=args.null_draws,
            targets_pp_spec=args.targets_pp,
        )
    except (pipeline.AnalysisError, ValueError) as exc:
        print(f"mdi analyze {args.what}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    meta = payload["meta"]
    print(f"mdi analyze {args.what} — exp {args.exp} ({_ANALYZE_FR[args.what]})")
    print(f"  store   : {store_path}")
    print(f"  records : {meta['n_records']} ({meta['n_scored']} scored)")
    print(f"  envs    : {', '.join(meta['env_ids'])}")
    print(f"  runs    : {', '.join(meta['run_ids'])}")
    print(f"  digest  : {meta['input_digest']}")
    print(f"  seed    : {meta['params']['seed']}")
    print(f"  wrote   : {path}")
    return 0


def _report(args: argparse.Namespace) -> int:
    """Regenerate paper artifacts deterministically from the derived tree (FR-014)."""
    derived_root = args.derived or os.path.join(args.data_dir, "derived")
    formats = [chunk.strip() for chunk in args.formats.split(",") if chunk.strip()]
    written: List[str] = []
    try:
        if args.target in ("figures", "all"):
            written.extend(
                pipeline.report_figures(
                    derived_root,
                    os.path.join(args.out, "figures"),
                    alpha=args.alpha,
                    formats=formats,
                )
            )
        if args.target in ("tables", "all"):
            written.extend(pipeline.report_tables(derived_root, os.path.join(args.out, "tables")))
        if args.target == "cost":
            cost_path = pipeline.report_cost_tex(derived_root, os.path.join(args.out, "tables"))
            if cost_path is not None:
                written.append(cost_path)
    except (pipeline.AnalysisError, ValueError) as exc:
        print(f"mdi report {args.target}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not written:
        print(
            f"mdi report {args.target}: nothing to render — no analysis payloads under "
            f"{derived_root}. Run `uv run mdi analyze <what> --exp <exp>` first.",
            file=sys.stderr,
        )
        return 1
    print(f"mdi report {args.target} — from {derived_root}")
    for path in sorted(written):
        print(f"  {path}")
    return 0


def _load_dotenv() -> None:
    """Load ``.env`` into the environment without overriding what is already set.

    AGENTS.md §3.4 puts secrets in ``.env`` or the environment; this makes the
    documented ``cp .env.example .env`` path actually work. Real environment
    variables win, so an exported key still overrides the file, and ``.env`` is
    gitignored so nothing here reaches a commit.
    """
    dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True), override=False)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point; returns the process exit status."""
    _load_dotenv()
    args = build_parser().parse_args(argv)
    if args.command == "estimate-cost":
        return _estimate_cost(args.config, args.data_dir)
    if args.command == "run":
        return _run(args.config, args.data_dir, args.resume, args.pilot)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "census":
        return _not_implemented(f"census {args.step}", _CENSUS_FR[args.step])
    if args.command == "report":
        return _report(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
