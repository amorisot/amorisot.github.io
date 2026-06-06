# prog-recon — Program Reconstruction Scaling Study

Engineering harness for the Program Reconstruction Scaling Study. This repo
implements the build specification in phases, gated by `# CHECKPOINT`s. The
companion scientific protocol governs *intent*; this code governs
*implementation*.

> **Status:** Vertical slice (Phases 1–8). All code runs **offline** with a
> `MockModel`. No real model API calls, no full-corpus generation, and no
> full grid enumeration have been performed. Those are gated behind
> CHECKPOINT 7 ("let it rip") and explicit cost sign-off.

See [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md) for the phase/checkpoint
status table and [the build spec](docs/build_spec.md) for the full plan.

## Quickstart

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
uv run pytest                 # all tests run offline with MockModel
uv run progrecon --help       # CLI: generate / run-slice / estimate / analyze
uv run progrecon dump-ops     # regenerate docs/op_semantics.md (CHECKPOINT 1 review doc)
uv run progrecon run-slice    # run ONE cheap cell end-to-end (offline oracle by default)
uv run progrecon estimate     # estimate spend and gate it against the ceiling (runs nothing)
```

To run against a real model (post-CHECKPOINT 7, deliberately): set
`models.cheap.provider: anthropic` in `config.yaml` and export `ANTHROPIC_API_KEY`.
The metered client aborts if cumulative spend would cross `budget.usd_ceiling`.

## Hard guardrails (enforced in code)

1. No provider SDK is called outside `runner/model_client.py`.
2. Every API call is metered and bounded by a global USD ceiling that
   **aborts** the run when breached (`runner/budget.py`).
3. The agent under test never sees `D_test` — enforced by disjoint objects,
   separate seed derivation, and hard assertions in the harness and scorer.
4. Candidate (model-authored) programs execute only inside a locked-down
   subprocess sandbox: no network, temp-only fs, CPU/mem/wall-clock caps.
5. `Transform.source` is **codegen'd from** `Transform.graph` — never
   hand-maintained in parallel.
6. Every random choice is seeded and recorded in the task/run manifest.

## Layout

```
src/progrecon/
  types.py          # all pydantic data contracts (single source of truth)
  dsl/              # ops table, executor, dataflow graph, codegen
  corpus/           # generator, perturbation, twins, identifiability, complexity
  data/             # P_x sampling + prompt serialization
  harness/          # agent loop, prompts, candidate sandbox
  runner/           # metered model client, budget, grid, cell runner
  scoring/          # behavioral equivalence + metrics
  analysis/         # log aggregation + (stubbed) statistical models
  cli.py            # typer CLI
tests/              # pytest; all offline
```
