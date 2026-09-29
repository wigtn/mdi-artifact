"""Async repeated-judging runner (FR-002): resume-safe, idempotent, cost-gated.

Call key ``(env_id, item_id, system_id, repeat_idx)`` is idempotent — a resumed
run skips already-completed calls with no double billing. Concurrency stays
within provider rate limits (per-provider backoff). Every run is preceded by
the cost gate (see :mod:`mdi.cost`; AGENTS.md §3.2) and writes raw records via
:mod:`mdi.store` only (AGENTS.md §3.1 — never ad-hoc scripts against judge
APIs).

Gate order (PRD §4.2, enforced here, not by convention):

1. ``mdi estimate-cost`` writes a receipt under ``data/gates/`` keyed by the
   canonical config hash. A run without a matching receipt raises
   :class:`EstimateRequired`.
2. ``mdi run --pilot`` executes the config's pilot slice and, on success,
   writes the pilot-completion flag.
3. A full run without that flag raises :class:`PilotRequired`.

Parse failures never trigger a re-prompt (ADR-011 D7). The runner keeps the
failed record (``parse_ok: false``) and redraws a *fresh* ``repeat_idx`` in the
same environment, until N valid repeats exist or the 2N attempt cap is hit.
"""

import asyncio
import contextlib
import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import yaml

from mdi import cost, parse, store
from mdi.grid import canonical_json, env_id, expand_grid, prompt_family_field
from mdi.providers import build_client
from mdi.providers.base import JudgeClient, JudgeRequest, ProviderError

__all__ = [
    "BudgetExceeded",
    "EstimateRequired",
    "PilotRequired",
    "ProviderError",
    "Paths",
    "RunPlan",
    "estimate",
    "load_config",
    "plan_run",
    "run",
    "write_estimate_receipt",
]

DEFAULT_DATA_DIR = "data"
DEFAULT_CONCURRENCY = 8
DEFAULT_MAX_OUTPUT_TOKENS = 64
SOURCE_PLACEHOLDER = "{{source}}"
CANDIDATE_PLACEHOLDER = "{{candidate}}"


class BudgetExceeded(Exception):
    """Estimated cost exceeds the configured cap (PRD §5.1)."""


class EstimateRequired(BudgetExceeded):
    """No ``mdi estimate-cost`` receipt on record for this config (PRD §4.2 gate order).

    A :class:`BudgetExceeded` subclass: "we do not know what this costs" is the
    degenerate case of "this may exceed the cap".
    """


class PilotRequired(Exception):
    """Full-grid run attempted before the pilot completion flag is set (PRD §5.1)."""


class Paths(NamedTuple):
    """Filesystem layout for one run (created lazily by the runner, never committed)."""

    data_dir: str
    raw_dir: str
    inputs_dir: str
    gates_dir: str
    ledger_path: str


class Paraphrase(NamedTuple):
    """One paraphrase variant of a cell: its id, rendered prompt and per-call provenance.

    A single-paraphrase (legacy) cell holds exactly one of these; an Exp 3
    paraphrase family holds one per paraphrase. Each carries its OWN
    ``payload_hash`` (the specific rendered prompt) so per-call provenance stays
    exact even though the whole family shares one ``env_id`` (ADR-009 / ADR-011 D1).
    """

    paraphrase_id: str
    prompt: str
    payload_hash: str
    in_tokens_est: int


class Cell(NamedTuple):
    """One (env, item, system) judging cell with its paraphrase variants.

    ``paraphrases`` holds one entry for a single-paraphrase config and the whole
    family (>= 2) for Exp 3; :func:`paraphrase_for` maps each repeat to one of
    them round-robin by ``repeat_idx``.
    """

    env_id: str
    env: Dict[str, Any]
    item_id: str
    system_id: str
    paraphrases: Tuple[Paraphrase, ...]
    out_tokens_est: int
    out_tokens_cap: int


class WorkItem(NamedTuple):
    """One scheduled judge call: a cell plus the repeat index and its seed.

    The paraphrase used is not stored — it is derived from ``cell`` and
    ``repeat_idx`` via :func:`paraphrase_for`, so scheduling, cost estimation and
    record emission all agree on the same deterministic round-robin.
    """

    cell: Cell
    repeat_idx: int
    seed: int


def paraphrase_for(cell: Cell, repeat_idx: int) -> Paraphrase:
    """The paraphrase a given repeat uses — round-robin by ``repeat_idx`` (deterministic).

    Repeat ``r`` uses ``cell.paraphrases[r % k]``. With N repeats and k
    paraphrases, N=20 / k=4 gives exactly 5 repeats per paraphrase; parse-failure
    redraws (``repeat_idx >= N``) simply continue the rotation, and a resumed run
    picks up at the next contiguous ``repeat_idx`` so the rotation is preserved.
    A single-paraphrase (legacy) cell always returns its one paraphrase — its
    records are byte-identical to the pre-family runner.
    """
    return cell.paraphrases[repeat_idx % len(cell.paraphrases)]


class RunPlan(NamedTuple):
    """Deterministic expansion of a config into cells, calls and provenance."""

    experiment: str
    config_hash: str
    pilot: bool
    n_repeats: int
    max_attempts: int
    cells: Tuple[Cell, ...]
    inputs_digest: str
    n_items: int
    system_ids: Tuple[str, ...]

    @property
    def expected_calls(self) -> List[cost.CallSpec]:
        """Call specs for the nominal N repeats per cell, at expected output size."""
        return self._calls(self.n_repeats, worst=False)

    @property
    def worst_case_calls(self) -> List[cost.CallSpec]:
        """Call specs at the 2N attempt cap and the output-token cap — the budget-gate bound."""
        return self._calls(self.max_attempts, worst=True)

    def _calls(self, repeats: int, worst: bool) -> List[cost.CallSpec]:
        """Expand cells into *repeats* call specs each.

        Expected uses the measured/typical output size; worst uses the hard
        output cap — so a reasoning model with a high safety cap does not
        inflate the expected column (recalibration, 2026-08-02). Each repeat is
        priced against the paraphrase it will actually use (:func:`paraphrase_for`),
        so a family whose paraphrases differ in length is estimated honestly.
        """
        specs: List[cost.CallSpec] = []
        for cell in self.cells:
            for repeat_idx in range(repeats):
                paraphrase = paraphrase_for(cell, repeat_idx)
                specs.append(
                    cost.CallSpec(
                        model=str(cell.env["judge_model"]),
                        in_tokens=paraphrase.in_tokens_est,
                        out_tokens=cell.out_tokens_cap if worst else cell.out_tokens_est,
                    )
                )
        return specs


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


def load_config(path: str) -> Dict[str, Any]:
    """Load an experiment YAML config."""
    with open(path, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: config must be a mapping")
    return loaded


def config_hash(config: Dict[str, Any]) -> str:
    """Canonical hash of the whole config — the key for gate receipts."""
    digest = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    return digest[:16]


def paths_for(data_dir: str = DEFAULT_DATA_DIR) -> Paths:
    """Derive the data-tree paths (directories are created on demand by writers)."""
    return Paths(
        data_dir=data_dir,
        raw_dir=os.path.join(data_dir, "raw"),
        inputs_dir=os.path.join(data_dir, "inputs"),
        gates_dir=os.path.join(data_dir, "gates"),
        ledger_path=os.path.join(data_dir, "ledger.jsonl"),
    )


def _runtime(config: Dict[str, Any]) -> Dict[str, Any]:
    """Runtime knobs (concurrency, token estimation) with defaults."""
    runtime = config.get("runtime") or {}
    max_output = int(runtime.get("max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS))
    return {
        "concurrency": int(runtime.get("concurrency", DEFAULT_CONCURRENCY)),
        "max_output_tokens": max_output,
        # Expected output tokens for the cost ESTIMATE. Defaults to the hard cap
        # (conservative); set from measured pilot usage to keep estimates honest
        # for reasoning models whose cap must stay high but whose typical output
        # is far below it. The worst-case column always uses the cap.
        "expected_output_tokens": int(runtime.get("expected_output_tokens", max_output)),
        "chars_per_token": float(runtime.get("chars_per_token", cost.DEFAULT_CHARS_PER_TOKEN)),
    }


def render_prompt(template: str, source: str, candidate: str) -> str:
    """Render a judge prompt.

    Placeholders are ``{{source}}`` and ``{{candidate}}`` and are substituted
    literally — templates routinely contain JSON braces, so ``str.format`` is
    not usable here.
    """
    for placeholder in (SOURCE_PLACEHOLDER, CANDIDATE_PLACEHOLDER):
        if placeholder not in template:
            raise ValueError(f"prompt template is missing the {placeholder} placeholder")
    return template.replace(SOURCE_PLACEHOLDER, source).replace(CANDIDATE_PLACEHOLDER, candidate)


def _templates(config: Dict[str, Any]) -> Dict[str, str]:
    """Per-scale prompt templates declared in the config."""
    prompt = config.get("prompt") or {}
    templates = prompt.get("templates") or {}
    if not isinstance(templates, dict) or not templates:
        raise ValueError("config.prompt.templates must map each scale to a template string")
    return {str(scale): str(text) for scale, text in templates.items()}


def _input_spec(config: Dict[str, Any], task: str) -> Dict[str, Any]:
    """Item-source spec for *task* (``inputs:`` directly, or ``inputs.by_task``)."""
    inputs = config.get("inputs") or {}
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("config.inputs is required")
    by_task = inputs.get("by_task")
    if isinstance(by_task, dict):
        if task not in by_task:
            raise ValueError(f"config.inputs.by_task has no entry for task {task!r}")
        spec = by_task[task]
        if not isinstance(spec, dict):
            raise ValueError(f"config.inputs.by_task[{task!r}] must be a mapping")
        return spec
    return inputs


def _env_axes(config: Dict[str, Any]) -> Dict[str, List[Any]]:
    """Environment axes declared under ``grid:`` (task / scale / temperature)."""
    grid = config.get("grid") or {}
    if not isinstance(grid, dict) or not grid:
        raise ValueError("config.grid is required")
    axes: Dict[str, List[Any]] = {}
    for name, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"config.grid.{name} must be a non-empty list")
        axes[str(name)] = list(values)
    return axes


def _env_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """Provider / judge env metadata shared by the single-paraphrase and family paths."""
    judge = config.get("judge") or {}
    prompt = config.get("prompt") or {}
    return {
        "benchmark": str(config.get("benchmark", "")),
        "judge_model": str(judge["model"]),
        "judge_model_version": str(judge.get("model_version", "")),
        "judge_tier": str(judge.get("tier", "api")),
        "provider": str(judge.get("provider", "")),
        "endpoint": str(judge.get("endpoint", "")),
        "serving_engine": judge.get("serving_engine"),
        "quantization": judge.get("quantization"),
        "prompt_id": str(prompt.get("prompt_id", "p_base")),
    }


def _env_cell(config: Dict[str, Any], axis_values: Dict[str, Any], template: str) -> Dict[str, Any]:
    """Assemble the single-paraphrase environment mapping that ``env_id`` hashes.

    Per ADR-009 §3 the prompt template *source text* (not its id) is part of the
    hash input; per ADR-006 Update the provider and endpoint are env metadata.
    This is the legacy path — a config with no ``prompt.paraphrases`` list keeps
    exactly this env mapping, and therefore its env_id, unchanged.
    """
    prompt = config.get("prompt") or {}
    env = _env_metadata(config)
    env["paraphrase_id"] = str(prompt.get("paraphrase_id", "pp_0"))
    env["prompt_template"] = template
    env.update(axis_values)
    return env


def _env_cell_family(
    config: Dict[str, Any],
    axis_values: Dict[str, Any],
    variants: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    """Assemble the environment mapping for a paraphrase FAMILY — one env, many paraphrases.

    Unlike :func:`_env_cell`, this hashes the whole family instead of one active
    template: ``prompt_family`` carries every paraphrase's template *source text*
    for this scale (ADR-009 §3 provenance intact) and ``prompt_family_id`` names
    the set. All paraphrases of the family therefore share ONE env_id, which is
    what makes the procedural floor omega^2 measurable within a single
    environment (ADR-011 D1). The single active paraphrase's template is
    deliberately NOT hashed here — the per-call paraphrase_id and its rendered
    prompt live on each ScoreRecord instead.
    """
    prompt = config.get("prompt") or {}
    env = _env_metadata(config)
    env["prompt_family_id"] = str(prompt.get("prompt_family_id", prompt.get("prompt_id", "p_base")))
    env["prompt_family"] = prompt_family_field(variants)
    env.update(axis_values)
    return env


def _paraphrase_family(config: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The declared paraphrase family (Exp 3), or ``None`` for a single-paraphrase config.

    A family is ``config.prompt.paraphrases``: a non-empty list of items
    ``{paraphrase_id: str, templates?: {scale: text}}`` (``template_overrides`` is
    accepted as an alias for the per-scale override map). A paraphrase whose
    override omits a scale falls back to the base ``prompt.templates`` for that
    scale — paraphrases vary only surface wording, never rubric content
    (ADR-011 D1). Returning ``None`` keeps the config on the legacy env_id.
    """
    prompt = config.get("prompt") or {}
    raw = prompt.get("paraphrases")
    if raw is None:
        return None
    if not isinstance(raw, list) or not raw:
        raise ValueError("config.prompt.paraphrases must be a non-empty list when declared")
    family: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or "paraphrase_id" not in item:
            raise ValueError("each config.prompt.paraphrases item needs a paraphrase_id")
        overrides = item.get("templates")
        alias = item.get("template_overrides")
        if overrides is not None and alias is not None:
            raise ValueError(
                f"paraphrase {item['paraphrase_id']!r}: set either templates or "
                "template_overrides, not both"
            )
        chosen = overrides if overrides is not None else alias
        if chosen is None:
            chosen = {}
        if not isinstance(chosen, dict):
            raise ValueError(
                f"paraphrase {item['paraphrase_id']!r}: templates must map scale -> text"
            )
        family.append(
            {
                "paraphrase_id": str(item["paraphrase_id"]),
                "templates": {str(scale): str(text) for scale, text in chosen.items()},
            }
        )
    return family


def _scale_variants(
    family: Optional[List[Dict[str, Any]]],
    config: Dict[str, Any],
    scale: str,
    templates: Dict[str, str],
) -> List[Tuple[str, str]]:
    """``(paraphrase_id, effective template)`` pairs for one scale, sorted by paraphrase_id.

    Legacy (no family): the single configured ``paraphrase_id`` with the scale's
    base template — identical to the pre-family behaviour, so the one Paraphrase
    it yields reproduces today's rendered prompt and payload hash exactly. Family:
    each paraphrase's override for this scale if present, else the base template.
    """
    base = templates[scale]
    if family is None:
        prompt = config.get("prompt") or {}
        return [(str(prompt.get("paraphrase_id", "pp_0")), base)]
    variants = [(item["paraphrase_id"], item["templates"].get(scale, base)) for item in family]
    variants.sort(key=lambda pair: pair[0])
    return variants


def _render_paraphrase(
    paraphrase_id: str,
    template: str,
    item: store.EvalItem,
    system_id: str,
    chars_per_token: float,
) -> Paraphrase:
    """Render one paraphrase's prompt for an (item, system) and derive its provenance."""
    prompt = render_prompt(template, item["source"], item["outputs"][system_id])
    return Paraphrase(
        paraphrase_id=paraphrase_id,
        prompt=prompt,
        payload_hash=store.request_payload_hash(prompt),
        in_tokens_est=cost.estimate_tokens(prompt, chars_per_token),
    )


# --------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------


def plan_run(
    config: Dict[str, Any],
    pilot: bool,
    data_dir: str = DEFAULT_DATA_DIR,
    allow_fetch: bool = True,
) -> RunPlan:
    """Expand a config into a deterministic set of judging cells.

    Materializes the item/system-output text into the content-addressed input
    snapshot store first (ADR-009 §2), so the plan — and every record derived
    from it — is anchored to exact bytes.
    """
    paths = paths_for(data_dir)
    runtime = _runtime(config)
    templates = _templates(config)
    family = _paraphrase_family(config)
    axes = _env_axes(config)
    repeats = config.get("repeats") or {}
    pilot_cfg = config.get("pilot") or {}

    n_repeats = int(pilot_cfg.get("n_repeats", 3)) if pilot else int(repeats.get("n", 5))
    factor = int(repeats.get("max_attempts_factor", parse.DEFAULT_MAX_ATTEMPTS_FACTOR))
    max_items = int(pilot_cfg["n_items"]) if pilot and "n_items" in pilot_cfg else None

    cells: List[Cell] = []
    item_digests: List[str] = []
    all_systems: List[str] = []
    n_items = 0

    for axis_values in expand_grid(axes):
        scale = str(axis_values["scale"])
        if scale not in templates:
            raise ValueError(f"config.prompt.templates has no template for scale {scale!r}")
        task = str(axis_values["task"])
        items = store.resolve_items(_input_spec(config, task), paths.inputs_dir, allow_fetch)
        if max_items is not None:
            items = items[:max_items]
        snapshot = store.materialize_items(paths.inputs_dir, items)
        item_digests.append(snapshot["inputs_digest"])
        all_systems.extend(snapshot["system_ids"])
        n_items = max(n_items, len(items))

        variants = _scale_variants(family, config, scale, templates)
        env = (
            _env_cell(config, axis_values, templates[scale])
            if family is None
            else _env_cell_family(config, axis_values, variants)
        )
        cell_env_id = env_id(env)
        for item in items:
            for system_id in sorted(item["outputs"]):
                paraphrases = tuple(
                    _render_paraphrase(
                        paraphrase_id, template, item, system_id, runtime["chars_per_token"]
                    )
                    for paraphrase_id, template in variants
                )
                cells.append(
                    Cell(
                        env_id=cell_env_id,
                        env=env,
                        item_id=item["item_id"],
                        system_id=system_id,
                        paraphrases=paraphrases,
                        out_tokens_est=runtime["expected_output_tokens"],
                        out_tokens_cap=runtime["max_output_tokens"],
                    )
                )

    cells.sort(key=lambda cell: (cell.env_id, cell.item_id, cell.system_id))
    inputs_digest = store.snapshot_json(paths.inputs_dir, sorted(set(item_digests)))
    return RunPlan(
        experiment=str(config.get("experiment", "experiment")),
        config_hash=config_hash(config),
        pilot=pilot,
        n_repeats=n_repeats,
        max_attempts=parse.max_attempts(n_repeats, factor),
        cells=tuple(cells),
        inputs_digest=inputs_digest,
        n_items=n_items,
        system_ids=tuple(sorted(set(all_systems))),
    )


def _seed_for(config: Dict[str, Any], repeat_idx: int) -> int:
    """Per-repeat seed (ADR-011 D1: seeds are recorded even where a provider ignores them)."""
    repeats = config.get("repeats") or {}
    return int(repeats.get("seed_base", 0)) + repeat_idx


# --------------------------------------------------------------------------------------
# Cost gate
# --------------------------------------------------------------------------------------


def estimate(
    config: Dict[str, Any],
    pilot: bool,
    data_dir: str = DEFAULT_DATA_DIR,
    allow_fetch: bool = True,
) -> Dict[str, Any]:
    """Dry-run estimate for one slice: expected (N) and worst-case (2N) cost + verdict."""
    plan = plan_run(config, pilot=pilot, data_dir=data_dir, allow_fetch=allow_fetch)
    paths = paths_for(data_dir)
    spent = cost.ledger_total_usd(paths.ledger_path)
    expected = cost.estimate_cost(config, plan.expected_calls)
    worst_case = cost.estimate_cost(config, plan.worst_case_calls)
    verdict = cost.budget_verdict(config, worst_case["cost_usd"], spent, pilot=pilot)
    return {
        "experiment": plan.experiment,
        "config_hash": plan.config_hash,
        "pilot": pilot,
        "cells": len(plan.cells),
        "items": plan.n_items,
        "systems": list(plan.system_ids),
        "n_repeats": plan.n_repeats,
        "max_attempts": plan.max_attempts,
        "inputs_digest": plan.inputs_digest,
        "expected": expected,
        "worst_case": worst_case,
        "verdict": verdict,
    }


def estimate_receipt_path(paths: Paths, hash_value: str) -> str:
    """Path of the ``estimate-cost`` receipt for a config hash."""
    return os.path.join(paths.gates_dir, f"{hash_value}.estimate.json")


def pilot_flag_path(paths: Paths, hash_value: str) -> str:
    """Path of the pilot-completion flag for a config hash."""
    return os.path.join(paths.gates_dir, f"{hash_value}.pilot.json")


def write_estimate_receipt(paths: Paths, hash_value: str, payload: Dict[str, Any]) -> str:
    """Record that ``estimate-cost`` ran for this exact config (PRD §4.2 gate order)."""
    path = estimate_receipt_path(paths, hash_value)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(payload))
    return path


def _require_gates(config: Dict[str, Any], plan: RunPlan, paths: Paths) -> None:
    """Enforce the estimator receipt and the pilot-before-full-grid order."""
    if not os.path.exists(estimate_receipt_path(paths, plan.config_hash)):
        raise EstimateRequired(
            f"no cost-gate receipt for config hash {plan.config_hash}; "
            "run `uv run mdi estimate-cost --config <path>` first (AGENTS.md §3.2)"
        )
    if not plan.pilot and not os.path.exists(pilot_flag_path(paths, plan.config_hash)):
        raise PilotRequired(
            f"no completed pilot for config hash {plan.config_hash}; "
            "run `uv run mdi run --config <path> --pilot` first (PRD §4.2)"
        )
    spent = cost.ledger_total_usd(paths.ledger_path)
    worst_case = cost.estimate_cost(config, plan.worst_case_calls)
    verdict = cost.budget_verdict(config, worst_case["cost_usd"], spent, pilot=plan.pilot)
    if verdict["verdict"] == cost.VERDICT_BLOCK:
        raise BudgetExceeded(verdict["reason"])


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


class _CellState:
    """Mutable per-cell progress: attempts spent and valid repeats collected."""

    def __init__(self, records: Sequence[store.ScoreRecord]) -> None:
        """Seed the state from records already in the raw store (resume)."""
        self.attempts = len(records)
        self.valid = sum(1 for record in records if record["parse_ok"])
        self.next_idx = max((record["repeat_idx"] for record in records), default=-1) + 1

    def remaining(self, n_repeats: int, max_attempts: int) -> int:
        """How many calls may still be issued for this cell."""
        return max(0, min(n_repeats - self.valid, max_attempts - self.attempts))

    def observe(self, parse_ok: bool) -> None:
        """Record the outcome of one issued call."""
        self.attempts += 1
        self.next_idx += 1
        if parse_ok:
            self.valid += 1


def _now_iso() -> str:
    """Current UTC timestamp, ISO-8601 with offset."""
    return datetime.now(timezone.utc).isoformat()


def _make_run_id(config_hash_value: str, now: Callable[[], str]) -> str:
    """Run id in the PRD §5.2 shape: ``r_<yyyymmdd>_<short hash of start time + config>``.

    The suffix mixes the microsecond-precision start timestamp into the config
    hash, so two runs of the same config on the same day stay distinguishable in
    the ledger.
    """
    stamp = now()
    date = stamp[:10].replace("-", "")
    suffix = hashlib.sha256(f"{stamp}|{config_hash_value}".encode("utf-8")).hexdigest()[:6]
    return f"r_{date}_{suffix}"


def _build_record(
    run_id: str,
    work: WorkItem,
    response_text: str,
    in_tokens: int,
    out_tokens: int,
    model_version: str,
    cost_usd: float,
    now: Callable[[], str],
) -> store.ScoreRecord:
    """Assemble one schema-v2 score record from a completed call."""
    env = work.cell.env
    paraphrase = paraphrase_for(work.cell, work.repeat_idx)
    result = parse.parse_response(response_text, str(env["scale"]))
    return store.ScoreRecord(
        schema_version=store.SCHEMA_VERSION,
        run_id=run_id,
        env_id=work.cell.env_id,
        judge_model=str(env["judge_model"]),
        judge_model_version=model_version,
        judge_tier=str(env["judge_tier"]),
        serving_engine=env["serving_engine"],
        quantization=env["quantization"],
        seed=work.seed,
        prompt_id=str(env["prompt_id"]),
        paraphrase_id=paraphrase.paraphrase_id,
        temperature=float(env["temperature"]),
        scale=str(env["scale"]),
        task=str(env["task"]),
        benchmark=str(env["benchmark"]),
        item_id=work.cell.item_id,
        system_id=work.cell.system_id,
        repeat_idx=work.repeat_idx,
        request_payload_hash=paraphrase.payload_hash,
        raw_response=response_text,
        parsed_score=result.score,
        parse_ok=result.ok,
        usage=store.Usage(in_tokens=in_tokens, out_tokens=out_tokens),
        cost_usd=cost_usd,
        ts=now(),
    )


class RunResult(NamedTuple):
    """Outcome of one run: id, realized spend and call/parse counts."""

    run_id: str
    calls: int
    parse_failures: int
    cost_usd: float
    in_tokens: int
    out_tokens: int


async def _execute(
    config: Dict[str, Any],
    plan: RunPlan,
    paths: Paths,
    client: JudgeClient,
    run_id: str,
    now: Callable[[], str],
) -> RunResult:
    """Run the wave loop: schedule, execute, redraw parse failures within the 2N cap."""
    runtime = _runtime(config)
    table = cost.price_table(config)
    limits = cost.caps(config)
    ledger_spent = cost.ledger_total_usd(paths.ledger_path)

    shards: Dict[str, str] = {
        cell.env_id: store.shard_path(paths.raw_dir, plan.experiment, cell.env_id)
        for cell in plan.cells
    }
    states: Dict[Tuple[str, str, str], _CellState] = {}
    for env_key, path in sorted(shards.items()):
        grouped = store.records_by_cell(store.load_shard(path).records)
        for cell in plan.cells:
            if cell.env_id != env_key:
                continue
            key = (cell.env_id, cell.item_id, cell.system_id)
            states[key] = _CellState(grouped.get(key, []))

    total_calls = 0
    total_failures = 0
    total_cost = 0.0
    total_in = 0
    total_out = 0
    semaphore = asyncio.Semaphore(runtime["concurrency"])

    with contextlib.ExitStack() as stack:
        writers = {
            env_key: stack.enter_context(store.ShardWriter(path))
            for env_key, path in sorted(shards.items())
        }
        write_locks = {env_key: asyncio.Lock() for env_key in writers}

        async def issue(work: WorkItem) -> store.ScoreRecord:
            """Issue one judge call and append its record to the shard."""
            request = JudgeRequest(
                model=str(work.cell.env["judge_model"]),
                prompt=paraphrase_for(work.cell, work.repeat_idx).prompt,
                temperature=float(work.cell.env["temperature"]),
                max_output_tokens=runtime["max_output_tokens"],
                seed=work.seed,
            )
            async with semaphore:
                response = await client.complete(request)
            call_cost = cost.price_call(
                table,
                str(work.cell.env["judge_model"]),
                response.in_tokens,
                response.out_tokens,
            )
            record = _build_record(
                run_id=run_id,
                work=work,
                response_text=response.text,
                in_tokens=response.in_tokens,
                out_tokens=response.out_tokens,
                model_version=response.model_version,
                cost_usd=round(call_cost, cost.USD_ROUNDING),
                now=now,
            )
            async with write_locks[work.cell.env_id]:
                writers[work.cell.env_id].append(record)
            return record

        for _wave in range(plan.max_attempts + 1):
            wave: List[WorkItem] = []
            for cell in plan.cells:
                state = states[(cell.env_id, cell.item_id, cell.system_id)]
                for offset in range(state.remaining(plan.n_repeats, plan.max_attempts)):
                    repeat_idx = state.next_idx + offset
                    wave.append(
                        WorkItem(
                            cell=cell,
                            repeat_idx=repeat_idx,
                            seed=_seed_for(config, repeat_idx),
                        )
                    )
            if not wave:
                break

            wave_estimate = cost.estimate_cost(
                config,
                [
                    cost.CallSpec(
                        model=str(w.cell.env["judge_model"]),
                        in_tokens=paraphrase_for(w.cell, w.repeat_idx).in_tokens_est,
                        out_tokens=w.cell.out_tokens_est,
                    )
                    for w in wave
                ],
            )
            projected = total_cost + wave_estimate["cost_usd"]
            if ledger_spent + projected > limits["cumulative_cap_usd"]:
                raise BudgetExceeded(
                    f"next wave would take cumulative spend to ${ledger_spent + projected:.4f}, "
                    f"over the cap ${limits['cumulative_cap_usd']:.2f}"
                )

            records = await asyncio.gather(*(issue(work) for work in wave))
            for record in records:
                key = (record["env_id"], record["item_id"], record["system_id"])
                states[key].observe(record["parse_ok"])
                total_calls += 1
                total_cost += record["cost_usd"]
                total_in += record["usage"]["in_tokens"]
                total_out += record["usage"]["out_tokens"]
                if not record["parse_ok"]:
                    total_failures += 1

    return RunResult(
        run_id=run_id,
        calls=total_calls,
        parse_failures=total_failures,
        cost_usd=round(total_cost, cost.USD_ROUNDING),
        in_tokens=total_in,
        out_tokens=total_out,
    )


def run(
    config: Dict[str, Any],
    resume: Optional[str] = None,
    pilot: bool = False,
    data_dir: str = DEFAULT_DATA_DIR,
    client: Optional[JudgeClient] = None,
    allow_fetch: bool = True,
    now: Optional[Callable[[], str]] = None,
) -> str:
    """Execute repeated judging for a config grid; returns the ``run_id``.

    ``pilot`` restricts execution to the config's pilot slice; ``resume``
    continues an interrupted run under its original ``run_id`` — resumption is
    idempotent on ``(env_id, item_id, system_id, repeat_idx)`` regardless, so
    completed calls are never re-billed.

    ``client`` is injectable so tests exercise the loop without a network; when
    omitted, the judge declared in the config is constructed and its credentials
    are read from the environment.
    """
    clock = now if now is not None else _now_iso
    paths = paths_for(data_dir)
    plan = plan_run(config, pilot=pilot, data_dir=data_dir, allow_fetch=allow_fetch)
    _require_gates(config, plan, paths)

    run_id = resume if resume else _make_run_id(plan.config_hash, clock)
    owns_client = client is None
    judge_client = (
        client
        if client is not None
        else build_client(config.get("judge") or {}, _runtime(config)["concurrency"])
    )

    async def main() -> RunResult:
        """Execute the plan and always release a runner-owned client."""
        try:
            return await _execute(config, plan, paths, judge_client, run_id, clock)
        finally:
            if owns_client:
                await judge_client.aclose()

    result = asyncio.run(main())

    cost.append_ledger_entry(
        paths.ledger_path,
        {
            "run_id": result.run_id,
            "experiment": plan.experiment,
            "config_hash": plan.config_hash,
            "inputs_digest": plan.inputs_digest,
            "pilot": plan.pilot,
            "calls": result.calls,
            "parse_failures": result.parse_failures,
            "in_tokens": result.in_tokens,
            "out_tokens": result.out_tokens,
            "cost_usd": result.cost_usd,
            "ts": clock(),
        },
    )

    if plan.pilot:
        path = pilot_flag_path(paths, plan.config_hash)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(
                canonical_json(
                    {
                        "run_id": result.run_id,
                        "config_hash": plan.config_hash,
                        "calls": result.calls,
                        "cost_usd": result.cost_usd,
                        "ts": clock(),
                    }
                )
            )
    return result.run_id
