"""Cost estimator + spend ledger (FR-004).

Design (AGENTS.md §3.2, PRD §4.2):

- **No paid API call without a cost gate.** Every run goes
  ``mdi estimate-cost`` -> pilot -> full grid; the runner blocks otherwise.
- Default hard caps pending ADR-003 kickoff decision: **pilot $50 /
  cumulative $100** — raisable via config only, never in code.
- Estimator: expand the config grid -> call count x token estimate x provider
  unit price -> per-model cost + OK/BLOCK verdict against remaining budget.
- Ledger: actual spend is appended per run to ``data/ledger.jsonl`` (runner
  only — AGENTS.md §6), accumulating tokens x unit price.
- Full-grid execution requires a completed-pilot flag (``PilotRequired``
  otherwise, PRD §5.1) — enforced in :mod:`mdi.runner`.

Prices are **declared in the config** (``pricing:``), never hardcoded here:
provider rates change, and an account-specific rate is not a property of the
codebase. A model with no declared price is an estimation error, not a zero.
"""

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypedDict

DEFAULT_PILOT_CAP_USD: float = 50.0
DEFAULT_CUMULATIVE_CAP_USD: float = 100.0

# Deterministic token heuristic for the dry run only; actual usage always comes
# from the provider response and is what the ledger records.
DEFAULT_CHARS_PER_TOKEN: float = 4.0
DEFAULT_PROMPT_OVERHEAD_TOKENS: int = 8

USD_ROUNDING: int = 6

VERDICT_OK = "OK"
VERDICT_BLOCK = "BLOCK"


class UnknownPriceError(Exception):
    """A model in the plan has no unit price declared in the config's price table."""


class CallSpec(TypedDict):
    """One planned judge call as the estimator sees it."""

    model: str
    in_tokens: int
    out_tokens: int


class ModelCost(TypedDict):
    """Per-model rollup of a cost estimate."""

    model: str
    calls: int
    in_tokens: int
    out_tokens: int
    cost_usd: float


class CostEstimate(TypedDict):
    """Dry-run estimate for a set of planned calls."""

    calls: int
    in_tokens: int
    out_tokens: int
    cost_usd: float
    by_model: List[ModelCost]


class BudgetVerdict(TypedDict):
    """OK/BLOCK judgment of an estimate against the configured caps."""

    verdict: str
    reason: str
    slice_cap_usd: float
    cumulative_cap_usd: float
    spent_usd: float
    projected_usd: float


def estimate_tokens(
    text: str,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    overhead_tokens: int = DEFAULT_PROMPT_OVERHEAD_TOKENS,
) -> int:
    """Deterministic character-based token estimate for the dry run."""
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    return int(len(text) / chars_per_token) + overhead_tokens


def price_table(config: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Read the config-declared unit price table (USD per million tokens)."""
    raw = config.get("pricing") or {}
    if not isinstance(raw, dict):
        raise ValueError("config.pricing must be a mapping of model -> unit prices")
    table: Dict[str, Dict[str, float]] = {}
    for model, entry in raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"pricing[{model!r}] must be a mapping")
        missing = {"input_usd_per_mtok", "output_usd_per_mtok"} - set(entry)
        if missing:
            raise ValueError(f"pricing[{model!r}] missing {sorted(missing)}")
        table[str(model)] = {
            "input_usd_per_mtok": float(entry["input_usd_per_mtok"]),
            "output_usd_per_mtok": float(entry["output_usd_per_mtok"]),
        }
    return table


def price_call(
    table: Dict[str, Dict[str, float]], model: str, in_tokens: int, out_tokens: int
) -> float:
    """USD cost of one call under the declared price table."""
    entry = table.get(model)
    if entry is None:
        raise UnknownPriceError(
            f"no price declared for model {model!r}; add it under `pricing:` in the config"
        )
    return (
        in_tokens * entry["input_usd_per_mtok"] + out_tokens * entry["output_usd_per_mtok"]
    ) / 1e6


def estimate_cost(config: Dict[str, Any], calls: Sequence[CallSpec]) -> CostEstimate:
    """Dry-run estimate for *calls*: totals plus a per-model rollup.

    The call list comes from :func:`mdi.runner.plan_run`, which expands the grid
    and renders every prompt — so the estimate is derived from the same work
    items the runner would execute, not from a parallel guess.
    """
    table = price_table(config)
    rollup: Dict[str, ModelCost] = {}
    for call in calls:
        model = call["model"]
        cost = price_call(table, model, call["in_tokens"], call["out_tokens"])
        bucket = rollup.setdefault(
            model,
            ModelCost(model=model, calls=0, in_tokens=0, out_tokens=0, cost_usd=0.0),
        )
        bucket["calls"] += 1
        bucket["in_tokens"] += call["in_tokens"]
        bucket["out_tokens"] += call["out_tokens"]
        bucket["cost_usd"] += cost
    by_model = [rollup[model] for model in sorted(rollup)]
    for bucket in by_model:
        bucket["cost_usd"] = round(bucket["cost_usd"], USD_ROUNDING)
    return CostEstimate(
        calls=sum(bucket["calls"] for bucket in by_model),
        in_tokens=sum(bucket["in_tokens"] for bucket in by_model),
        out_tokens=sum(bucket["out_tokens"] for bucket in by_model),
        cost_usd=round(sum(bucket["cost_usd"] for bucket in by_model), USD_ROUNDING),
        by_model=by_model,
    )


def caps(config: Dict[str, Any]) -> Dict[str, float]:
    """Budget caps for this config — PRD §4.2 defaults unless raised in the config."""
    budget = config.get("budget") or {}
    return {
        "pilot_cap_usd": float(budget.get("pilot_cap_usd", DEFAULT_PILOT_CAP_USD)),
        "cumulative_cap_usd": float(budget.get("cumulative_cap_usd", DEFAULT_CUMULATIVE_CAP_USD)),
    }


def budget_verdict(
    config: Dict[str, Any], projected_usd: float, spent_usd: float, pilot: bool
) -> BudgetVerdict:
    """Judge a projected spend against the pilot and cumulative caps.

    The pilot cap bounds a single pilot slice; the cumulative cap bounds
    *ledger to date + this run* and applies to every run, pilot or not.
    """
    limits = caps(config)
    slice_cap = limits["pilot_cap_usd"] if pilot else limits["cumulative_cap_usd"]
    cumulative_cap = limits["cumulative_cap_usd"]
    projected = round(projected_usd, USD_ROUNDING)
    spent = round(spent_usd, USD_ROUNDING)

    reason = "within caps"
    verdict = VERDICT_OK
    if projected > slice_cap:
        verdict = VERDICT_BLOCK
        label = "pilot" if pilot else "run"
        reason = f"projected ${projected:.4f} exceeds the {label} cap ${slice_cap:.2f}"
    elif spent + projected > cumulative_cap:
        verdict = VERDICT_BLOCK
        reason = (
            f"ledger ${spent:.4f} + projected ${projected:.4f} "
            f"exceeds the cumulative cap ${cumulative_cap:.2f}"
        )
    return BudgetVerdict(
        verdict=verdict,
        reason=reason,
        slice_cap_usd=slice_cap,
        cumulative_cap_usd=cumulative_cap,
        spent_usd=spent,
        projected_usd=projected,
    )


# --------------------------------------------------------------------------------------
# Spend ledger (data/ledger.jsonl) — append-only, runner-written (AGENTS.md §6)
# --------------------------------------------------------------------------------------


def append_ledger_entry(ledger_path: str, entry: Dict[str, Any]) -> None:
    """Append one actual-spend entry to the ledger (append-only, runner-only)."""
    if "cost_usd" not in entry:
        raise ValueError("ledger entry must carry cost_usd")
    parent = os.path.dirname(os.path.abspath(ledger_path))
    os.makedirs(parent, exist_ok=True)
    line = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    with open(ledger_path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_ledger(ledger_path: str) -> List[Dict[str, Any]]:
    """Read every ledger entry; a truncated final line is skipped, not fatal."""
    if not os.path.exists(ledger_path):
        return []
    entries: List[Dict[str, Any]] = []
    with open(ledger_path, "r", encoding="utf-8") as fh:
        lines = [line for line in fh.read().split("\n") if line.strip()]
    for index, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                continue
            raise
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def ledger_total_usd(ledger_path: str, experiment: Optional[str] = None) -> float:
    """Cumulative recorded spend, optionally restricted to one experiment."""
    return round(
        sum(
            float(entry.get("cost_usd", 0.0))
            for entry in read_ledger(ledger_path)
            if experiment is None or entry.get("experiment") == experiment
        ),
        USD_ROUNDING,
    )


def total_actual_cost(
    table: Dict[str, Dict[str, float]], usages: Iterable[Dict[str, Any]]
) -> float:
    """Sum the priced cost of realized calls (``{model, in_tokens, out_tokens}``)."""
    return round(
        sum(
            price_call(table, str(u["model"]), int(u["in_tokens"]), int(u["out_tokens"]))
            for u in usages
        ),
        USD_ROUNDING,
    )
