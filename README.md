# MDI: The Minimum Detectable Improvement of an LLM Judge

Code, data and manuscript source for *MDI: The Minimum Detectable Improvement
of an LLM Judge* (Hyeong-seob Kim, WIGTN), presented at the TAE (Trust-AI-Eval)
workshop at NeurIPS 2026. Every number in the paper resolves to a file in this
repository.

## What is here

| Path | What |
|---|---|
| `src/mdi/`, `tests/` | The framework. Every estimator in the paper lives here. |
| `configs/` | One YAML per experiment. The registry's config hash is the first 16 hex chars of SHA-256 over the canonicalized parsed config: `uv run python -c "from mdi import runner; print(runner.config_hash(runner.load_config('configs/exp1_dense.yaml')))"` reproduces the registry row without side effects. (`mdi estimate-cost` prints it too, but re-writes this config's gate receipt under `data/gates/`; the receipts committed here are the ones the original runs cleared.) |
| `data/raw/` | Every judge call ever made, append-only, one JSONL record per scoring. 97,752 records. That is the registry's 95,522 full-grid calls plus 1,490 pilot passes and 740 spike probes, all rows of `data/ledger.jsonl`. |
| `data/inputs/` | Content-addressed snapshots of the benchmark items, so a re-run scores the same text. |
| `data/derived/` | Analysis outputs. Every number in the paper resolves to one of these files. |
| `data/ledger.jsonl` | Per-run cost and call counts, the source of the run registry table. |
| `data/gates/` | The estimate-cost and pilot records each run had to clear before the full grid. |
| `paper/figures/`, `paper/tables/` | Generated artifacts, never hand-edited. |
| `paper/tae2026/` | The manuscript source. |

## Reproducing the paper

Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync

# Regenerate every figure and table from the derived records.
uv run mdi report all

# Re-derive those records from the raw scores. No API calls, no cost.
# The flags matter: several analyses default to a narrower sweep than the
# paper used, and every derived file records the parameters it was built with
# under `meta.params`, so you can check any of these against the committed
# output.
uv run mdi analyze variance  --exp exp1_dense
uv run mdi analyze decay     --exp exp3_decay8
uv run mdi analyze fip       --exp exp1_dense --n-repeats 1,5,10
uv run mdi analyze mdi-table --exp exp1_dense --sweep 1,3,5,8,10 --targets-pp 0.5,1,2

# The remaining Table 1 rows come from the sibling grids: repeat with
# --exp exp1_anchor, exp1_anchor_scales, exp1_alpaca, exp1_ladder.
```

Each writes into `data/derived/<exp>/`, overwriting the committed copy with a
byte-identical one. If a file comes out different, diff its `meta.params`
against the committed version first; that is where a wrong flag shows up.

`mdi report all` run twice produces byte-identical output; that is the
determinism claim in the appendix, and it is checkable here:

```bash
uv run mdi report all && cp -r paper/figures /tmp/a && cp -r paper/tables /tmp/b
uv run mdi report all && diff -r /tmp/a paper/figures && diff -r /tmp/b paper/tables
```

Byte-identical is a same-platform claim: the PDFs and tables are stable
across platforms, but PNG bytes re-encode under each platform's zlib while
the pixels stay identical, so compare rasters pixel-wise when crossing
platforms.

Re-issuing the judge calls themselves needs an OpenAI key and money, and goes
through the cost gate the paper describes:

```bash
uv run mdi estimate-cost --config configs/exp3_decay8.yaml
uv run mdi run --config configs/exp3_decay8.yaml --pilot
uv run mdi run --config configs/exp3_decay8.yaml
```

The runner is idempotent on `(env_id, item_id, system_id, repeat_idx)`, so a
resumed run bills only the calls it has not made.

## Tracing a number

Every quantity in the paper resolves to a derived file plus a `run_id`. The
figure and table comments in the manuscript source name both. For example the
noise floor of 8.48 %p is `omega` in `data/derived/exp3_decay8/decay.json`;
its `meta.run_ids` names the three gated passes behind the 16,000 records it
was fit on: the 120-call pilot `r_20260805_6ecb21`, then `r_20260805_a9d7a3`
(9,480 calls) and `r_20260805_523960` (6,400 calls, config hash
`07f436c4afaa217e`), all three rows of `data/ledger.jsonl`; the registry
table carries the two full passes.

## Browsing the raw scores

`data/raw/` holds one JSONL shard per environment, some of them large. If a
directory listing fails to expand, the files open directly:

- `data/raw/exp1_dense/`: the judge-variance grid
- `data/raw/exp3_decay8/e_e941458fd533.jsonl`: the eight-paraphrase sweep (16,000 records)

## Citation

```bibtex
@inproceedings{kim2026mdi,
  title     = {{MDI}: The Minimum Detectable Improvement of an {LLM} Judge},
  author    = {Kim, Hyeong-seob},
  booktitle = {TAE (Trust-AI-Eval): Can We Trust AI Evaluation? Workshop at NeurIPS 2026},
  year      = {2026},
  note      = {Non-archival}
}
```

## License

Code (`src/`, `tests/`, `configs/`) is released under the
[Apache License 2.0](LICENSE). The data, figures, tables and manuscript text
this project produced (`data/raw/`, `data/derived/`, `data/ledger.jsonl`,
`data/gates/`, `paper/`) are released under
[CC BY 4.0](LICENSE-DATA). Benchmark items under `data/inputs/` (SummEval,
AlpacaEval 2.0) and the Regimes release used in Experiment 4 remain under their
original licenses, and `paper/tae2026/neurips_2026.sty` belongs to NeurIPS.
