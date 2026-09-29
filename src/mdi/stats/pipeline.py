"""Analysis and report drivers behind ``mdi analyze`` / ``mdi report`` (FR-005..FR-009, FR-014).

This is the only place the analysis layer touches the filesystem. It reads the
**raw store** (via :mod:`mdi.store` — never a private JSONL reader), writes
derived artifacts under ``data/derived/<exp>/``, and regenerates
``paper/figures/`` and ``paper/tables/`` from those derived files. Nothing here
calls a provider or mutates ``data/raw/``.

Determinism (AGENTS.md §3.3): derived JSON is written with sorted keys, ASCII
escapes and LF line endings; tables use fixed-precision formatting and sorted
rows; figures go through :mod:`mdi.stats.figures`, which pins
``SOURCE_DATE_EPOCH`` and file metadata. No wall-clock, hostname or absolute
path is ever embedded — so ``mdi report all`` twice is byte-identical.

Provenance (AGENTS.md §3.6): every derived payload carries the schema version,
the sorted ``run_id`` list, an input digest over the exact records consumed, and
the full parameter set of the analysis. Figures carry the same identifiers in a
footer line.
"""

import csv
import hashlib
import io
import json
import os
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    cast,
)

from mdi import store
from mdi.stats import decay as decay_mod
from mdi.stats import figures as figures_mod
from mdi.stats import fip as fip_mod
from mdi.stats import mdi_table as mdi_mod
from mdi.stats import null_mdi as null_mod
from mdi.stats import oracle as oracle_mod
from mdi.stats import records as rec
from mdi.stats import variance as var_mod
from mdi.stats.bootstrap import DEFAULT_CI_LEVEL, DEFAULT_N_RESAMPLES, DEFAULT_SEED
from mdi.stats.scales import maybe_pp, scale_width
from mdi.store import ScoreRecord

ANALYSIS_FILES: Dict[str, str] = {
    "variance": "variance.json",
    "fip": "fip.json",
    "decay": "decay.json",
    "mdi-table": "mdi_table.json",
    "promotions": "promotions.json",
    "oracle": "oracle.json",
}

DEFAULT_SCREENING_REPEATS: str = "0"
"""ADR-007 §1: the one-pass screening run is held out of FIP estimation."""

FLOAT_FORMAT: str = "{:.6g}"


class AnalysisError(RuntimeError):
    """A user-facing analysis failure (missing store, unusable data, bad parameters)."""


# --------------------------------------------------------------------------------------
# Store I/O
# --------------------------------------------------------------------------------------


def default_store_path(data_dir: str, exp: str) -> str:
    """Raw store location for an experiment: ``<data_dir>/raw/<exp>`` (a shard directory)."""
    return os.path.join(data_dir, "raw", exp)


def default_derived_dir(data_dir: str, exp: str) -> str:
    """Derived output location for an experiment: ``<data_dir>/derived/<exp>``."""
    return os.path.join(data_dir, "derived", exp)


def shard_paths(store_path: str) -> List[str]:
    """Sorted shard files for *store_path* (a shard file, or a directory of shards)."""
    if os.path.isdir(store_path):
        return sorted(
            os.path.join(store_path, name)
            for name in os.listdir(store_path)
            if name.endswith(".jsonl") and not name.endswith(".quarantine.jsonl")
        )
    return [store_path]


def load_store(store_path: str) -> List[ScoreRecord]:
    """Load every record of a raw store through :func:`mdi.store.read_records`."""
    if not os.path.exists(store_path):
        raise AnalysisError(
            f"raw store not found: {store_path} — run `uv run mdi run --config <cfg>` first, "
            "or point --store at an existing shard file or directory"
        )
    paths = shard_paths(store_path)
    if not paths:
        raise AnalysisError(f"no .jsonl shards under {store_path}")
    records: List[ScoreRecord] = []
    for path in paths:
        records.extend(store.read_records(path))
    if not records:
        raise AnalysisError(f"raw store holds no records: {store_path}")
    return rec.sorted_records(records)


def write_json(path: str, payload: Dict[str, Any]) -> str:
    """Write *payload* deterministically (sorted keys, ASCII, LF, trailing newline)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def read_json(path: str) -> Dict[str, Any]:
    """Read one derived JSON payload."""
    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return cast(Dict[str, Any], loaded)


# --------------------------------------------------------------------------------------
# Parameter parsing
# --------------------------------------------------------------------------------------


def parse_int_spec(spec: str) -> List[int]:
    """Parse ``"0"``, ``"1,3,5"`` or ``"0-4,7"`` into a sorted list of ints (``""`` -> empty)."""
    values: List[int] = []
    for chunk in spec.split(","):
        piece = chunk.strip()
        if not piece:
            continue
        if "-" in piece[1:]:
            head, _, tail = piece.partition("-")
            start, end = int(head), int(tail)
            if end < start:
                raise AnalysisError(f"invalid range {piece!r}")
            values.extend(range(start, end + 1))
        else:
            values.append(int(piece))
    return sorted(set(values))


def parse_float_spec(spec: str) -> List[float]:
    """Parse a comma-separated float list (e.g. ``"0.10,0.05,0.01"``)."""
    out = [float(chunk) for chunk in spec.split(",") if chunk.strip()]
    if not out:
        raise AnalysisError(f"expected at least one number in {spec!r}")
    return out


def observed_repeats(records: Sequence[ScoreRecord]) -> List[int]:
    """Sorted ``repeat_idx`` values present in *records*."""
    return sorted({record["repeat_idx"] for record in records})


def resolve_split(
    records: Sequence[ScoreRecord],
    screening_spec: str,
    estimation_spec: Optional[str],
) -> fip_mod.RepeatSplit:
    """Build the ADR-007 repeat split from CLI specs, defaulting estimation to "the rest"."""
    screening = parse_int_spec(screening_spec)
    if estimation_spec:
        estimation = parse_int_spec(estimation_spec)
    else:
        estimation = [idx for idx in observed_repeats(records) if idx not in set(screening)]
    if not estimation:
        raise AnalysisError(
            "no estimation repeats left after holding out the screening repeats "
            f"{screening} (ADR-007) — the store has repeats {observed_repeats(records)}"
        )
    return fip_mod.make_split(screening=screening, estimation=estimation)


# --------------------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------------------


def provenance_meta(
    analysis: str,
    exp: str,
    records: Sequence[ScoreRecord],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Traceability block embedded in every derived payload (AGENTS.md §3.6)."""
    scored = rec.scored_records(records)
    return {
        "analysis": analysis,
        "exp": exp,
        "schema_version": store.SCHEMA_VERSION,
        "n_records": len(records),
        "n_scored": len(scored),
        "run_ids": rec.run_ids(records),
        "env_ids": rec.env_ids(records),
        "input_digest": rec.records_digest(records),
        "params": params,
    }


def analyze(
    what: str,
    exp: str,
    records: Sequence[ScoreRecord],
    *,
    env_ids: Optional[Sequence[str]] = None,
    seed: int = DEFAULT_SEED,
    screening_spec: str = DEFAULT_SCREENING_REPEATS,
    estimation_spec: Optional[str] = None,
    n_repeats_spec: str = "1",
    sweep_spec: str = "1,3,5,8,10,20",
    alphas_spec: str = "0.10,0.05,0.01",
    draws: Optional[int] = None,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    group_field: str = decay_mod.DEFAULT_GROUP_FIELD,
    alpha_level: str = var_mod.DEFAULT_ALPHA_LEVEL,
    readoff: str = mdi_mod.DEFAULT_READOFF,
    null_draws: Optional[int] = None,
    targets_pp_spec: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one analysis over *records* and return the derived payload (pure, no I/O)."""
    targets = sorted(env_ids) if env_ids else rec.env_ids(records)
    missing = sorted(set(targets) - set(rec.env_ids(records)))
    if missing:
        raise AnalysisError(f"env_id(s) not present in the store: {missing}")

    params: Dict[str, Any] = {
        "seed": seed,
        "env_ids": targets,
        "ci_level": ci_level,
        "n_resamples": n_resamples,
    }

    if what == "variance":
        params["alpha_level"] = alpha_level
        payload: Dict[str, Any] = dict(
            var_mod.variance_report(_subset(records, targets), alpha_level=alpha_level, seed=seed)
        )
    elif what == "fip":
        split = resolve_split(records, screening_spec, estimation_spec)
        params.update(
            {
                "screening_repeats": split["screening"],
                "estimation_repeats": split["estimation"],
                "n_repeats_sweep": parse_int_spec(n_repeats_spec),
                "draws_per_pair": draws or fip_mod.DEFAULT_DRAWS_PER_PAIR,
                "sigma_bin_multiples": list(fip_mod.DEFAULT_SIGMA_BIN_MULTIPLES),
            }
        )
        payload = dict(
            fip_mod.fip_report(
                records,
                split=split,
                env_ids=targets,
                n_repeats_sweep=parse_int_spec(n_repeats_spec) or [1],
                seed=seed,
                draws_per_pair=draws or fip_mod.DEFAULT_DRAWS_PER_PAIR,
                n_resamples=n_resamples,
                ci_level=ci_level,
            )
        )
    elif what == "decay":
        params.update(
            {
                "sweep": parse_int_spec(sweep_spec),
                "group_field": group_field,
                "draws_per_cell": draws or decay_mod.DEFAULT_DRAWS_PER_CELL,
            }
        )
        payload = dict(
            decay_mod.decay_report(
                records,
                env_ids=targets,
                sweep=parse_int_spec(sweep_spec) or list(decay_mod.DEFAULT_REPEAT_SWEEP),
                group_field=group_field,
                seed=seed,
                draws_per_cell=draws or decay_mod.DEFAULT_DRAWS_PER_CELL,
                n_resamples=n_resamples,
                ci_level=ci_level,
            )
        )
    elif what == "mdi-table":
        split = resolve_split(records, screening_spec, estimation_spec)
        targets_pp = (
            parse_float_spec(targets_pp_spec)
            if targets_pp_spec is not None
            else list(mdi_mod.DEFAULT_TARGETS_PP)
        )
        params.update(
            {
                "screening_repeats": split["screening"],
                "estimation_repeats": split["estimation"],
                "sweep": parse_int_spec(sweep_spec),
                "alphas": parse_float_spec(alphas_spec),
                "draws_per_pair": draws or fip_mod.DEFAULT_DRAWS_PER_PAIR,
                "draws_per_system": null_draws or null_mod.DEFAULT_DRAWS_PER_SYSTEM,
                "readoff": readoff,
                "min_bin_draws": mdi_mod.DEFAULT_MIN_BIN_DRAWS,
                "targets_pp": targets_pp,
                "primary": "null_pi_half_width",
                "validation": "fip_crossing",
            }
        )
        payload = dict(
            mdi_mod.build_mdi_table(
                records,
                split=split,
                env_ids=targets,
                sweep=parse_int_spec(sweep_spec) or list(decay_mod.DEFAULT_REPEAT_SWEEP),
                alphas=parse_float_spec(alphas_spec),
                seed=seed,
                readoff=readoff,
                draws_per_pair=draws or fip_mod.DEFAULT_DRAWS_PER_PAIR,
                draws_per_system=null_draws or null_mod.DEFAULT_DRAWS_PER_SYSTEM,
                n_resamples=n_resamples,
                ci_level=ci_level,
                targets_pp=targets_pp,
            )
        )
    else:
        raise AnalysisError(f"unknown analysis: {what!r}")

    payload["meta"] = provenance_meta(what, exp, records, params)
    return payload


def _null_thresholds(
    table: Dict[str, Any],
    env_id: str,
    alpha: float,
) -> Dict[int, float]:
    """``{N: MDI_null(env, N, alpha)}`` in raw score units, from a derived MDI table."""
    out: Dict[int, float] = {}
    for entry in table.get("null_entries", []):
        if entry["env_id"] != env_id or abs(entry["alpha"] - alpha) > 1e-12:
            continue
        out[int(entry["n_repeats"])] = float(entry["mdi_null"])
    return out


def oracle_analysis(
    exp: str,
    *,
    store_path: str,
    out_dir: str,
    inputs_dir: str,
    facets: Sequence[str],
    env_ids: Optional[Sequence[str]] = None,
    seed: int = DEFAULT_SEED,
    screening_spec: str = DEFAULT_SCREENING_REPEATS,
    estimation_spec: Optional[str] = None,
    alpha: float = mdi_mod.DEFAULT_ALPHA,
    draws: Optional[int] = None,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> Dict[str, Any]:
    """External-oracle validation for every summarization env in *exp* (ADR-029).

    Three inputs, none of them new measurements: the raw store, the pinned
    annotation snapshot, and this experiment's derived MDI table (for the
    thresholds the claim rates are read against). Environments whose task is not
    summarization are skipped — the instruction-following axis has no external
    annotation, and saying so is part of the result (ADR-029 Consequences).
    """
    records = load_store(store_path)
    targets = sorted(env_ids) if env_ids else rec.env_ids(records)
    split = resolve_split(records, screening_spec, estimation_spec)
    payload_path = os.path.join(out_dir, ANALYSIS_FILES["mdi-table"])
    table = read_json(payload_path)
    annotations = oracle_mod.load_summeval_oracle(inputs_dir, facets=facets)
    # Per-facet dicts for the construct check (ADR-033): each facet re-read as if
    # it were the whole reference. Same pinned snapshot, no fetch; skipped when a
    # single facet was requested, where the breakdown would restate the average.
    annotations_by_facet = (
        {
            facet: oracle_mod.load_summeval_oracle(inputs_dir, facets=[facet])
            for facet in sorted(facets)
        }
        if len(facets) >= 2
        else None
    )

    reports: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for env_id in targets:
        env_records = rec.filter_records(records, env_id=env_id)
        if not env_records:
            continue
        meta = {m["env_id"]: m for m in rec.env_metadata(env_records)}[env_id]
        if meta["task"] != "summarization":
            skipped.append({"env_id": env_id, "task": meta["task"], "reason": "no external oracle"})
            continue
        thresholds = _null_thresholds(table, env_id, alpha)
        if not thresholds:
            skipped.append({"env_id": env_id, "task": meta["task"], "reason": "no MDI entries"})
            continue
        reports.append(
            dict(
                oracle_mod.build_oracle_report(
                    env_records,
                    env_id,
                    annotations,
                    split=split,
                    thresholds=thresholds,
                    alpha=alpha,
                    facets=facets,
                    oracle_by_facet=annotations_by_facet,
                    seed=seed,
                    draws_per_pair=draws or oracle_mod.DEFAULT_DRAWS_PER_PAIR,
                    n_resamples=n_resamples,
                    ci_level=ci_level,
                )
            )
        )
    if not reports:
        raise AnalysisError(
            f"exp {exp!r}: no summarization environment with an MDI table — "
            "the external oracle covers the summarization axis only"
        )
    payload: Dict[str, Any] = {
        "reports": reports,
        "skipped": skipped,
        "bias": {
            report["env_id"]: oracle_mod.bias_decomposition(report["gaps"]) for report in reports
        },
    }
    params: Dict[str, Any] = {
        "seed": seed,
        "env_ids": targets,
        "ci_level": ci_level,
        "n_resamples": n_resamples,
        "alpha": alpha,
        "facets": sorted(facets),
        "draws_per_pair": draws or oracle_mod.DEFAULT_DRAWS_PER_PAIR,
        "screening_repeats": split["screening"],
        "estimation_repeats": split["estimation"],
        "source": store.SUMMEVAL_SOURCE_SPEC,
        "mdi_table": os.path.basename(payload_path),
    }
    payload["meta"] = provenance_meta("oracle", exp, records, params)
    return payload


def _subset(records: Sequence[ScoreRecord], env_ids: Sequence[str]) -> List[ScoreRecord]:
    """Records restricted to *env_ids*, canonical order preserved."""
    wanted = set(env_ids)
    return [record for record in records if record["env_id"] in wanted]


def run_analysis(
    what: str,
    exp: str,
    *,
    store_path: str,
    out_dir: str,
    **kwargs: Any,
) -> Tuple[str, Dict[str, Any]]:
    """Load the store, run one analysis, and write ``<out_dir>/<analysis>.json``."""
    if what not in ANALYSIS_FILES:
        raise AnalysisError(f"unknown analysis: {what!r}")
    records = load_store(store_path)
    payload = analyze(what, exp, records, **kwargs)
    path = write_json(os.path.join(out_dir, ANALYSIS_FILES[what]), payload)
    return path, payload


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def derived_experiments(derived_root: str) -> List[Tuple[str, str]]:
    """Sorted ``(exp, dir)`` pairs under the derived root."""
    if not os.path.isdir(derived_root):
        return []
    return sorted(
        (name, os.path.join(derived_root, name))
        for name in os.listdir(derived_root)
        if os.path.isdir(os.path.join(derived_root, name))
    )


def _provenance_line(exp: str, meta: Dict[str, Any], env_id: str) -> str:
    """One-line provenance stamp for a figure footer."""
    digest = str(meta.get("input_digest", ""))
    runs = ",".join(str(value) for value in meta.get("run_ids", []))
    return f"exp {exp} · env {env_id} · runs {runs} · input {digest[:23]}"


def _null_mdi_index(
    payload: Dict[str, Any],
    alpha: float,
) -> Tuple[Dict[Tuple[str, int], float], Dict[str, List[null_mod.NullMdiEntry]]]:
    """Index a mdi-table payload's null entries at *alpha*: marker lookup + per-env series."""
    lookup: Dict[Tuple[str, int], float] = {}
    series: Dict[str, List[null_mod.NullMdiEntry]] = {}
    for entry_obj in payload.get("null_entries", []):
        entry = cast(null_mod.NullMdiEntry, entry_obj)
        if round(entry["alpha"], 9) != round(alpha, 9):
            continue
        lookup[(entry["env_id"], entry["n_repeats"])] = entry["mdi_null"]
        series.setdefault(entry["env_id"], []).append(entry)
    return lookup, series


def report_figures(
    derived_root: str,
    figures_dir: str,
    *,
    alpha: float = mdi_mod.DEFAULT_ALPHA,
    formats: Sequence[str] = figures_mod.DEFAULT_FORMATS,
) -> List[str]:
    """Regenerate every figure derivable from the derived tree (deterministic).

    FIP figures are labelled with the primary null-PI MDI (ADR-015a) joined
    from the experiment's ``mdi_table.json`` when it exists; the FIP-crossing
    validation value goes into the caption (renderer rule c). Each env with
    null entries at *alpha* also gets an MDI-vs-N figure (renderer rule b).
    """
    written: List[str] = []
    overlay_pool: Dict[Tuple[str, str], Tuple[fip_mod.FipCurve, Optional[float], str]] = {}
    teaser_pool: Dict[Tuple[str, str], List[null_mod.NullMdiEntry]] = {}
    for exp, path in derived_experiments(derived_root):
        null_lookup: Dict[Tuple[str, int], float] = {}
        null_series: Dict[str, List[null_mod.NullMdiEntry]] = {}
        mdi_meta: Dict[str, Any] = {}
        mdi_path = os.path.join(path, ANALYSIS_FILES["mdi-table"])
        if os.path.exists(mdi_path):
            mdi_payload = read_json(mdi_path)
            mdi_meta = cast(Dict[str, Any], mdi_payload.get("meta", {}))
            null_lookup, null_series = _null_mdi_index(mdi_payload, alpha)
        fip_path = os.path.join(path, ANALYSIS_FILES["fip"])
        if os.path.exists(fip_path):
            payload = read_json(fip_path)
            meta = cast(Dict[str, Any], payload.get("meta", {}))
            for curve_obj in payload.get("curves", []):
                curve = cast(fip_mod.FipCurve, curve_obj)
                if curve["n_repeats"] == 1:
                    for pool_exp, pool_env, pool_label in figures_mod.HEADLINE_FIP_ENVS:
                        if pool_exp == exp and pool_env == curve["env_id"]:
                            overlay_pool[(pool_exp, pool_env)] = (
                                curve,
                                null_lookup.get((curve["env_id"], 1)),
                                pool_label,
                            )
                written.extend(
                    figures_mod.render_fip_curve(
                        curve,
                        figures_dir,
                        stem=f"fip_curve_{exp}_{curve['env_id']}_N{curve['n_repeats']}",
                        alpha=alpha,
                        null_mdi=null_lookup.get((curve["env_id"], curve["n_repeats"])),
                        formats=formats,
                        provenance=_provenance_line(exp, meta, curve["env_id"]),
                    )
                )
        for env_id in sorted(null_series):
            for teaser_exp, teaser_env, _ in figures_mod.TEASER_MDI_ENVS:
                if teaser_exp == exp and teaser_env == env_id:
                    teaser_pool[(teaser_exp, teaser_env)] = null_series[env_id]
            written.extend(
                figures_mod.render_mdi_curve(
                    null_series[env_id],
                    figures_dir,
                    stem=f"mdi_null_curve_{exp}_{env_id}",
                    formats=formats,
                    provenance=_provenance_line(exp, mdi_meta, env_id),
                )
            )
        decay_path = os.path.join(path, ANALYSIS_FILES["decay"])
        if os.path.exists(decay_path):
            payload = read_json(decay_path)
            meta = cast(Dict[str, Any], payload.get("meta", {}))
            for fit_obj in payload.get("fits", []):
                fit = cast(decay_mod.DecayFit, fit_obj)
                written.extend(
                    figures_mod.render_decay_curve(
                        fit,
                        figures_dir,
                        stem=f"decay_curve_{exp}_{fit['env_id']}",
                        formats=formats,
                        provenance=_provenance_line(exp, meta, fit["env_id"]),
                    )
                )
        promotions_path = os.path.join(path, ANALYSIS_FILES["promotions"])
        if os.path.exists(promotions_path):
            payload = read_json(promotions_path)
            meta = cast(Dict[str, Any], payload.get("meta", {}))
            rows_all = cast(List[Dict[str, Any]], payload.get("rows", []))
            for stop_obj in payload.get("stopping", []):
                stop = cast(Dict[str, Any], stop_obj)
                seed = stop["seed"]
                seed_rows = [row for row in rows_all if row["seed"] == seed]
                if not seed_rows:
                    continue
                written.extend(
                    figures_mod.render_promotion_stopping(
                        seed_rows,
                        stop,
                        figures_dir,
                        stem=f"{exp}_stopping_seed{seed}",
                        alpha=float(payload.get("alpha", alpha)),
                        formats=formats,
                        # Manuscript figure: no visible provenance stamp (same
                        # rule as the overlays). Run ids live in the LaTeX
                        # source comment and in promotions.json meta.
                        provenance=None,
                    )
                )
    # Manuscript overview figure: MDI(env, N) for every Experiment 1 environment.
    teaser_series = [
        (teaser_pool[(exp, env)], label)
        for exp, env, label in figures_mod.TEASER_MDI_ENVS
        if (exp, env) in teaser_pool
    ]
    if teaser_series:
        written.extend(
            figures_mod.render_mdi_overlay(
                teaser_series,
                figures_dir,
                stem=f"mdi_overlay_alpha{alpha:g}".replace(".", "p"),
                formats=formats,
                # No visible provenance stamp: this figure is the manuscript's
                # opening exhibit and a 5 pt grey line under the x-axis label
                # read as dirt. Traceability lives in the LaTeX source comment
                # beside the \includegraphics, in TEASER_MDI_ENVS, and in the
                # PDF metadata; the per-env diagnostic renders keep theirs.
                provenance=None,
            )
        )
    # Manuscript FIP figure: the headline environments on one axes. Built from the
    # same curve objects the per-env panels use, so the two can never disagree.
    overlay_series = [
        overlay_pool[(exp, env)]
        for exp, env, _ in figures_mod.HEADLINE_FIP_ENVS
        if (exp, env) in overlay_pool
    ]
    if overlay_series:
        written.extend(
            figures_mod.render_fip_overlay(
                overlay_series,
                figures_dir,
                stem=f"fip_overlay_N1_alpha{alpha:g}".replace(".", "p"),
                alpha=alpha,
                formats=formats,
                provenance=None,
            )
        )
    return sorted(written)


def _format_cell(value: Any) -> str:
    """Deterministic cell rendering: fixed-precision floats, empty string for ``None``."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return FLOAT_FORMAT.format(value)
    return str(value)


def write_csv(path: str, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Write a CSV with LF terminators and fixed float formatting."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(header))
    for row in rows:
        writer.writerow([_format_cell(value) for value in row])
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(buffer.getvalue())
    return path


def _collect(
    derived_root: str,
    analysis: str,
    key: str,
    builder: Callable[[str, Dict[str, Any]], List[List[Any]]],
) -> List[List[Any]]:
    """Gather table rows for *analysis* across every experiment in the derived tree."""
    rows: List[List[Any]] = []
    for exp, path in derived_experiments(derived_root):
        file_path = os.path.join(path, ANALYSIS_FILES[analysis])
        if not os.path.exists(file_path):
            continue
        payload = read_json(file_path)
        for entry in payload.get(key, []):
            rows.extend(builder(exp, cast(Dict[str, Any], entry)))
    return rows


def report_tables(derived_root: str, tables_dir: str) -> List[str]:
    """Regenerate every paper table from the derived tree (deterministic, sorted)."""
    written: List[str] = []

    variance_rows = _collect(
        derived_root,
        "variance",
        "envs",
        lambda exp, entry: [
            [
                exp,
                entry["env_id"],
                entry["task"],
                entry["scale"],
                entry["judge_model"],
                entry["temperature"],
                entry["sigma"],
                entry.get("sigma_ci_lo"),
                entry.get("sigma_ci_hi"),
                entry["flip_rate"],
                entry["krippendorff_alpha"],
                entry["alpha_level"],
                entry["n_cells"],
                entry["n_scores"],
                entry["parse_failure_rate"],
            ]
        ],
    )
    if variance_rows:
        written.append(
            write_csv(
                os.path.join(tables_dir, "variance_by_env.csv"),
                [
                    "exp",
                    "env_id",
                    "task",
                    "scale",
                    "judge_model",
                    "temperature",
                    "sigma",
                    "sigma_ci_lo",
                    "sigma_ci_hi",
                    "flip_rate",
                    "krippendorff_alpha",
                    "alpha_level",
                    "n_cells",
                    "n_scores",
                    "parse_failure_rate",
                ],
                sorted(variance_rows, key=_row_key),
            )
        )

    conversion_rows = _collect(
        derived_root,
        "fip",
        "conversion_table",
        lambda exp, entry: [
            [
                exp,
                entry["env_id"],
                entry["task"],
                entry["scale"],
                entry["n_repeats"],
                entry.get("delta_lo_pp", maybe_pp(entry["delta_lo"], entry["scale"])),
                entry.get("delta_hi_pp", maybe_pp(entry["delta_hi"], entry["scale"])),
                entry.get("delta_pp", maybe_pp(entry["delta"], entry["scale"])),
                entry["reversal_pct"],
                entry["ci_lo_pct"],
                entry["ci_hi_pct"],
                entry["n_draws"],
                entry["delta_lo"],
                entry["delta_hi"],
                entry["delta"],
                entry["statement"],
            ]
        ],
    )
    if conversion_rows:
        written.append(
            write_csv(
                os.path.join(tables_dir, "fip_conversion.csv"),
                [
                    "exp",
                    "env_id",
                    "task",
                    "scale",
                    "n_repeats",
                    "delta_lo_pp",
                    "delta_hi_pp",
                    "delta_pp",
                    "reversal_pct",
                    "ci_lo_pct",
                    "ci_hi_pct",
                    "n_draws",
                    "delta_lo",
                    "delta_hi",
                    "delta",
                    "statement",
                ],
                sorted(conversion_rows, key=_row_key),
            )
        )

    mdi_rows: List[List[Any]] = []
    for exp, path in derived_experiments(derived_root):
        file_path = os.path.join(path, ANALYSIS_FILES["mdi-table"])
        if not os.path.exists(file_path):
            continue
        mdi_rows.extend(_mdi_table_rows(exp, read_json(file_path)))
    if mdi_rows:
        written.append(
            write_csv(
                os.path.join(tables_dir, "mdi_table.csv"),
                [
                    "exp",
                    "task",
                    "scale",
                    "env_id",
                    "n_repeats",
                    "alpha",
                    "mdi_null_pp",
                    "mdi_null_smoothed_pp",
                    "mdi_null_ci_lo_pp",
                    "mdi_null_ci_hi_pp",
                    "power",
                    "mdi_power_pp",
                    "ratio_power",
                    "pool_inflation",
                    "mdi_null_corrected_pp",
                    "mdi_fip_pp",
                    "ratio_fip_over_null",
                    "abs_diff_pp",
                    "mdi_null",
                    "mdi_fip",
                    "attained_fip",
                    "sigma",
                    "readoff",
                    "n_draws_null",
                    "n_draws_fip",
                ],
                sorted(mdi_rows, key=_row_key),
            )
        )

    # Both readings of the inversion land in one file, separated by a `power`
    # column (ADR-024a). The alpha-level rows carry power 0.5 — a true gain of
    # exactly the target clears that bar about half the time — so a reader who
    # sorts on `power` sees immediately which question each row answers.
    def _reverse_row(exp: str, entry: Dict[str, Any]) -> List[List[Any]]:
        return [
            [
                exp,
                entry["task"],
                entry["scale"],
                entry["env_id"],
                entry["alpha"],
                entry.get("power", 0.5),
                entry["target_pp"],
                entry["recommended_n"] if entry["reachable"] else "unreachable",
                entry.get("mdi_at_recommended_pp"),
                entry["target_raw"],
            ]
        ]

    reverse_rows = _collect(derived_root, "mdi-table", "reverse_entries", _reverse_row)
    reverse_rows.extend(
        _collect(derived_root, "mdi-table", "reverse_entries_at_power", _reverse_row)
    )
    if reverse_rows:
        written.append(
            write_csv(
                os.path.join(tables_dir, "recommended_repeats.csv"),
                [
                    "exp",
                    "task",
                    "scale",
                    "env_id",
                    "alpha",
                    "power",
                    "target_improvement_pp",
                    "recommended_n",
                    "mdi_at_recommended_pp",
                    "target_improvement_raw",
                ],
                sorted(reverse_rows, key=_row_key),
            )
        )

    decay_rows = _collect(
        derived_root,
        "decay",
        "fits",
        lambda exp, entry: [
            [
                exp,
                entry["env_id"],
                entry["task"],
                entry["scale"],
                point["n"],
                point["var_mean"],
                maybe_pp(point["sd_mean"], entry["scale"]),
                maybe_pp(point["sd_theoretical"], entry["scale"]),
                point["sd_mean"],
                point["sd_theoretical"],
                point["deviation"],
                entry["sigma_sq"],
                entry["omega_sq"],
                entry.get("omega_sq_ci_lo"),
                entry.get("omega_sq_ci_hi"),
                entry["omega_sq_debiased"],
                entry["icc"],
                entry["r_squared"],
                figures_mod.noise_floor_display(cast(decay_mod.DecayFit, entry)),
            ]
            for point in entry["points"]
        ],
    )
    if decay_rows:
        written.append(
            write_csv(
                os.path.join(tables_dir, "decay_curve.csv"),
                [
                    "exp",
                    "env_id",
                    "task",
                    "scale",
                    "n_repeats",
                    "var_mean",
                    "sd_mean_pp",
                    "sd_theoretical_pp",
                    "sd_mean",
                    "sd_theoretical",
                    "deviation",
                    "sigma_sq",
                    "omega_sq",
                    "omega_sq_ci_lo",
                    "omega_sq_ci_hi",
                    "omega_sq_debiased",
                    "icc",
                    "r_squared",
                    "noise_floor_display",
                ],
                sorted(decay_rows, key=_row_key),
            )
        )

    sla_path = report_sla_margin_tex(derived_root, tables_dir)
    if sla_path is not None:
        written.append(sla_path)
    cost_path = report_cost_tex(derived_root, tables_dir)
    if cost_path is not None:
        written.append(cost_path)

    return sorted(written)


# --------------------------------------------------------------------------------------
# Paper-ready .tex tables: SLA margin (ADR-015 Update 2026-08-02) + cost (FR-012)
# --------------------------------------------------------------------------------------

SLA_MARGIN_ALPHA: float = 0.10
"""Null entries at this two-sided alpha carry ``abs_delta_quantile`` at the 0.90
level — by symmetry of the null delta this is the ONE-SIDED 95% margin: the
amount by which two independent scorings of the same system exceed each other
no more than 5% of the time (ADR-015 Update 2026-08-02)."""

PUBLISHED_EXPERIMENTS: FrozenSet[str] = frozenset(
    {"exp1_alpaca", "exp1_anchor", "exp1_anchor_scales", "exp1_dense"}
)
"""The measurement grids whose rows appear in RENDERED paper tables (ADR-032).

Rendered tables (``sla_margin.tex``, ``cost_table.tex``) go into the manuscript as
rows and numbers, so what feeds them has to be a deliberate list. The CSV
inventories are NOT filtered by this: they carry an ``exp`` column and their
consumers select, which is how ``spike_fip`` has always sat in ``mdi_table.csv``
without reaching the paper.

Membership is deliberately explicit rather than a name prefix. The prefix
heuristic it replaces (``startswith("spike")``) broke the moment a validation run
arrived that was not a spike: ``exp1_ladder`` (ADR-029 §e) scores the SAME
environment as ``exp1_dense`` by design — ``env_id`` hashes the environment and
not the system set — so it silently displaced the published dense row, adding a
duplicate line to the appendix SLA table and moving the cost table's per-cell
figures. Adding a grid here is now an act, not a side effect of running it.
"""

SLA_COLUMNS: Tuple[int, ...] = (1, 3, 5, 8)
COST_COLUMNS: Tuple[int, ...] = (1, 3, 8, 20)
COST_CELLS_BASIS: int = 100
"""Cost cells are USD per this many (item x system) cells — one 100-item
evaluation of a single system, matching the paper's gate framing."""

_TASK_DISPLAY: Dict[str, str] = {
    "summarization": "SummEval",
    "instruction_following": "AlpacaEval",
}
_TIER_BY_JUDGE: Dict[str, str] = {"gpt-4o-mini": "dense", "gpt-5.6": "anchor"}
_TIER_ORDER: Dict[str, int] = {"dense": 0, "anchor": 1}
#: Table-1 (body) naming scheme, so the generated SLA table needs no
#: reader-side translation between naming systems.
_JUDGE_SHORT: Dict[str, str] = {"gpt-4o-mini": "4o-mini", "gpt-5.6": "gpt-5.6"}
_TASK_SHORT: Dict[str, str] = {
    "summarization": "Summ",
    "instruction_following": "Instr",
}
_SCALE_SHORT: Dict[str, str] = {
    "likert5": "L5",
    "likert10": "L10",
    "score100": "S100",
}


def _write_text(path: str, text: str) -> str:
    """Write *text* with LF endings (deterministic byte-for-byte)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path


def _judge_by_env(derived_root: str) -> Dict[Tuple[str, str], str]:
    """Map ``(exp, env_id) -> judge_model`` from every variance payload."""
    out: Dict[Tuple[str, str], str] = {}
    for exp, path in derived_experiments(derived_root):
        file_path = os.path.join(path, ANALYSIS_FILES["variance"])
        if not os.path.exists(file_path):
            continue
        for entry_obj in read_json(file_path).get("envs", []):
            entry = cast(Dict[str, Any], entry_obj)
            out[(exp, str(entry["env_id"]))] = str(entry["judge_model"])
    return out


def _sigma_by_env(derived_root: str) -> Dict[Tuple[str, str], float]:
    """Map ``(exp, env_id) -> raw sigma`` from the variance payloads.

    The SLA margin's sigma-relative parentheses divide by THIS sigma — the
    all-repeat estimate the body's Table 1 reports — never by the null
    entry's estimation-half sigma, which drifts slightly on the anchor rows
    (7.02 vs 7.15 %p on likert5).
    """
    out: Dict[Tuple[str, str], float] = {}
    for exp, path in derived_experiments(derived_root):
        file_path = os.path.join(path, ANALYSIS_FILES["variance"])
        if not os.path.exists(file_path):
            continue
        for entry_obj in read_json(file_path).get("envs", []):
            entry = cast(Dict[str, Any], entry_obj)
            out[(exp, str(entry["env_id"]))] = float(entry["sigma"])
    return out


def report_sla_margin_tex(
    derived_root: str,
    tables_dir: str,
    *,
    published: FrozenSet[str] = PUBLISHED_EXPERIMENTS,
) -> Optional[str]:
    """Render ``sla_margin.tex``: the one-sided 95% absolute-claim margin per (env, N).

    A pure read-off of recorded quantities (ADR-015 Update 2026-08-02): the
    margin is ``abs_delta_quantile_pp`` of the alpha=0.10 null entry — the 0.90
    quantile of ``|delta_null|``, which by symmetry is the one-sided 95%
    quantile of the null delta. To claim "true score >= t", the observed mean
    must exceed ``t`` by at least this margin. Only
    :data:`PUBLISHED_EXPERIMENTS` feed this table (ADR-032); returns ``None``
    when no eligible entry exists.
    """
    judge_by_env = _judge_by_env(derived_root)
    sigma_by_env = _sigma_by_env(derived_root)
    rows: List[Tuple[Tuple[int, float, int, str], str, Dict[int, str]]] = []
    provenance: List[str] = []
    for exp, path in derived_experiments(derived_root):
        if exp not in published:
            continue
        file_path = os.path.join(path, ANALYSIS_FILES["mdi-table"])
        if not os.path.exists(file_path):
            continue
        payload = read_json(file_path)
        meta = cast(Dict[str, Any], payload.get("meta", {}))
        per_env: Dict[str, Dict[int, str]] = {}
        env_info: Dict[str, Tuple[str, str, float]] = {}
        for entry_obj in payload.get("null_entries", []):
            entry = cast(Dict[str, Any], entry_obj)
            if abs(float(entry["alpha"]) - SLA_MARGIN_ALPHA) > 1e-12:
                continue
            n_repeats = int(entry["n_repeats"])
            if n_repeats not in SLA_COLUMNS:
                continue
            env_id = str(entry["env_id"])
            margin_pp = float(entry["abs_delta_quantile_pp"])
            # sigma-relative parentheses divide by the SAME sigma the body's
            # Table 1 reports (all-repeat, variance payload) — not the null
            # entry's estimation-half sigma, which drifts on the anchor rows.
            sigma = sigma_by_env.get((exp, env_id))
            cell = f"{margin_pp:.2f}"
            if sigma:
                cell += f" ({float(entry['abs_delta_quantile']) / float(sigma):.2f}$\\sigma$)"
            per_env.setdefault(env_id, {})[n_repeats] = cell
            env_info[env_id] = (
                str(entry["task"]),
                str(entry["scale"]),
                scale_width(str(entry["scale"])),
            )
        if not per_env:
            continue
        run_ids = ", ".join(str(r) for r in meta.get("run_ids", []))
        provenance.append(f"%   {exp}: run_ids [{run_ids}]  {meta.get('input_digest', '?')}")
        for env_id in sorted(per_env):
            task, scale, width = env_info[env_id]
            judge = judge_by_env.get((exp, env_id), "?")
            tier = _TIER_ORDER.get(_TIER_BY_JUDGE.get(judge, judge), 9)
            # Table-1 naming and Table-1 row order (task block, then scale
            # width, dense before anchor) so readers translate nothing.
            judge_short = _JUDGE_SHORT.get(judge, judge)
            task_short = _TASK_SHORT.get(task, task)
            scale_short = _SCALE_SHORT.get(scale, scale)
            label = f"{judge_short} $\\cdot$ {task_short} $\\cdot$ {scale_short}  % {env_id}"
            task_order = 0 if task == "summarization" else 1
            sort_key = (task_order, width, tier, env_id)
            rows.append((sort_key, label, per_env[env_id]))
    if not rows:
        return None
    rows.sort(key=lambda item: item[0])
    lines: List[str] = [
        "% sla_margin.tex — generated by `mdi report tables`; DO NOT hand-edit.",
        "% One-sided 95% absolute-claim margin in %p: claim `score >= t` only when",
        "% the observed N-repeat mean exceeds t by at least the cell value",
        "% (sigma-relative in parentheses). Read-off of the alpha=0.10 null entry's",
        "% |delta| 0.90-quantile (decision record 015, 2026-08-02).",
        "% Provenance:",
        *provenance,
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l" + "c" * len(SLA_COLUMNS) + "}",
        "  \\toprule",
        "  & \\multicolumn{" + str(len(SLA_COLUMNS)) + "}{c}{Margin over $t$ [\\%p]} \\\\",
        "  \\cmidrule(lr){2-" + str(len(SLA_COLUMNS) + 1) + "}",
        "  Environment & " + " & ".join(f"$N{{=}}{n}$" for n in SLA_COLUMNS) + " \\\\",
        "  \\midrule",
    ]
    for _, label, cells in rows:
        text, env_comment = label.split("  % ")
        body = " & ".join(cells.get(n, "--") for n in SLA_COLUMNS)
        lines.append(f"  {text} & {body} \\\\  % {env_comment}")
    lines.extend(["  \\bottomrule", "\\end{tabular*}", ""])
    return _write_text(os.path.join(tables_dir, "sla_margin.tex"), "\n".join(lines))


def report_cost_tex(
    derived_root: str,
    tables_dir: str,
    ledger_path: Optional[str] = None,
    *,
    published: FrozenSet[str] = PUBLISHED_EXPERIMENTS,
) -> Optional[str]:
    """Render ``cost_table.tex``: measured USD per 100 cells at each repeat budget.

    Money math, sourced from the run ledger only (AGENTS.md §3.2/§3.6): per-call
    cost is the ledger's total spend over total calls per judge tier, where a
    ledger row maps to a tier through the derived variance payload of its
    experiment (rows whose experiment has no derived payload — e.g. one-off
    spikes — are skipped and named in a comment). Cell value = per-call cost x N
    x :data:`COST_CELLS_BASIS`. Returns ``None`` when the ledger is absent.
    """
    if ledger_path is None:
        ledger_path = os.path.join(os.path.dirname(os.path.abspath(derived_root)), "ledger.jsonl")
    if not os.path.exists(ledger_path):
        return None
    judge_by_exp: Dict[str, str] = {}
    for (exp, _env_id), judge in sorted(_judge_by_env(derived_root).items()):
        # ADR-032: only the published grids set a tier. A validation run on an
        # already-published environment would otherwise fold its calls into that
        # tier's measured per-call cost and move a number in the manuscript.
        if exp not in published:
            continue
        judge_by_exp.setdefault(exp, judge)
    with open(ledger_path, "rb") as handle:
        ledger_bytes = handle.read()
    totals: Dict[str, Tuple[int, float]] = {}
    skipped: List[str] = []
    for line in ledger_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = cast(Dict[str, Any], json.loads(line))
        exp = str(row["experiment"])
        row_judge: Optional[str] = judge_by_exp.get(exp)
        if row_judge is None:
            if exp not in skipped:
                skipped.append(exp)
            continue
        tier = _TIER_BY_JUDGE.get(row_judge, row_judge)
        calls, cost = totals.get(tier, (0, 0.0))
        totals[tier] = (calls + int(row["calls"]), cost + float(row["cost_usd"]))
    if not totals:
        return None
    digest = hashlib.sha256(ledger_bytes).hexdigest()[:12]
    lines = [
        "% cost_table.tex — generated by `mdi report tables`; DO NOT hand-edit.",
        f"% Measured USD per {COST_CELLS_BASIS} (item x system) cells at N scoring",
        "% repeats: ledger total spend / total calls per judge tier, x N x "
        + str(COST_CELLS_BASIS)
        + ".",
        f"% Provenance: data/ledger.jsonl sha256:{digest}",
    ]
    for tier in sorted(totals, key=lambda t: (_TIER_ORDER.get(t, 9), t)):
        calls, cost = totals[tier]
        lines.append(f"%   {tier}: {calls} calls, ${cost:.4f} logged")
    if skipped:
        lines.append("%   skipped (no derived tier mapping): " + ", ".join(sorted(skipped)))
    lines.extend(
        [
            "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l"
            + "c" * len(COST_COLUMNS)
            + "}",
            "  \\toprule",
            "  & \\multicolumn{"
            + str(len(COST_COLUMNS))
            + "}{c}{USD per "
            + str(COST_CELLS_BASIS)
            + " cells} \\\\",
            "  \\cmidrule(lr){2-" + str(len(COST_COLUMNS) + 1) + "}",
            "  Judge tier & " + " & ".join(f"$N{{=}}{n}$" for n in COST_COLUMNS) + " \\\\",
            "  \\midrule",
        ]
    )
    for tier in sorted(totals, key=lambda t: (_TIER_ORDER.get(t, 9), t)):
        calls, cost = totals[tier]
        per_call = cost / calls
        cells = " & ".join(f"{per_call * n * COST_CELLS_BASIS:.3f}" for n in COST_COLUMNS)
        lines.append(f"  {tier} & {cells} \\\\")
    lines.extend(["  \\bottomrule", "\\end{tabular*}", ""])
    return _write_text(os.path.join(tables_dir, "cost_table.tex"), "\n".join(lines))


def _mdi_table_rows(exp: str, payload: Dict[str, Any]) -> List[List[Any]]:
    """One mdi_table.csv row per (env, N, alpha): null-PI primary, FIP-crossing validation.

    Keys present on one path only still yield a row with the other side empty
    (e.g. sweep values the estimation repeats cannot support carry no null
    entry) — the table reports, it never filters (ADR-015b).
    """
    null_index: Dict[Tuple[str, int, float], Dict[str, Any]] = {
        (e["env_id"], e["n_repeats"], e["alpha"]): cast(Dict[str, Any], e)
        for e in payload.get("null_entries", [])
    }
    fip_index: Dict[Tuple[str, int, float], Dict[str, Any]] = {
        (e["env_id"], e["n_repeats"], e["alpha"]): cast(Dict[str, Any], e)
        for e in payload.get("entries", [])
    }
    rows: List[List[Any]] = []
    for key in set(null_index) | set(fip_index):
        null_entry = null_index.get(key)
        fip_entry = fip_index.get(key)
        head = null_entry if null_entry is not None else fip_entry
        assert head is not None
        scale = str(head["scale"])
        mdi_null = None if null_entry is None else null_entry["mdi_null"]
        mdi_fip = None if fip_entry is None else fip_entry["mdi"]
        ratio: Optional[float] = None
        abs_diff_pp: Optional[float] = None
        if mdi_null is not None and mdi_fip is not None:
            abs_diff_pp = maybe_pp(abs(mdi_null - mdi_fip), scale)
            if mdi_null != 0.0:
                ratio = mdi_fip / mdi_null
        rows.append(
            [
                exp,
                head["task"],
                scale,
                head["env_id"],
                head["n_repeats"],
                head["alpha"],
                None if null_entry is None else null_entry["mdi_null_pp"],
                None if null_entry is None else null_entry["mdi_null_smoothed_pp"],
                None if null_entry is None else null_entry["ci_lo_pp"],
                None if null_entry is None else null_entry["ci_hi_pp"],
                # ADR-024a (power) and ADR-025a (finite-pool correction). Both
                # sit beside the headline value rather than replacing it: the
                # published table is the alpha-level, uncorrected column, and
                # these say by how much each reading would move it.
                None if null_entry is None else null_entry.get("power"),
                None if null_entry is None else null_entry.get("mdi_power_pp"),
                None if null_entry is None else null_entry.get("ratio_power"),
                None if null_entry is None else null_entry.get("pool_inflation"),
                None if null_entry is None else null_entry.get("mdi_null_corrected_pp"),
                maybe_pp(mdi_fip, scale),
                ratio,
                abs_diff_pp,
                mdi_null,
                mdi_fip,
                None if fip_entry is None else fip_entry["attained"],
                head["sigma"],
                payload.get("readoff"),
                None if null_entry is None else null_entry["n_draws"],
                None if fip_entry is None else fip_entry["n_draws"],
            ]
        )
    return rows


def _cell_key(value: Any) -> Tuple[int, float, str]:
    """Type-aware sort key: ``None`` first, then numbers numerically, then text."""
    if value is None:
        return (0, 0.0, "")
    if isinstance(value, bool):
        return (1, float(value), "")
    if isinstance(value, (int, float)):
        return (2, float(value), "")
    return (3, 0.0, str(value))


def _row_key(row: Sequence[Any]) -> Tuple[Tuple[int, float, str], ...]:
    """Total order over table rows independent of value types."""
    return tuple(_cell_key(value) for value in row)
