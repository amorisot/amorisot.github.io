# Checkpoint status

The build proceeds in phases gated by `# CHECKPOINT`s. **Hard guardrails** that
were never crossed in this build:

- ❌ No real model API calls (default `cheap` provider is `mock`).
- ❌ No full-corpus generation (no 100 transforms; only small slices).
- ❌ No full grid enumeration / sweeps run.
- ❌ The `usd_ceiling` was never raised; `budget.estimate` was never skipped.

All tests run **offline** with `MockModel`.

| Phase | Output | Checkpoint | Code state | Human sign-off |
|------:|--------|------------|------------|----------------|
| 1 | `types.py` + config surface | **CHECKPOINT 0** — review contracts | ✅ built, tests pass | ⬜ pending |
| 2 | DSL ops + semantics table | **CHECKPOINT 1** (most critical) — review op semantics + 100% op tests | ✅ built, tests pass | ⬜ pending |
| 3 | executor + graph + sandbox | **CHECKPOINT 2** — executor/sandbox tests | ✅ built, tests pass | ⬜ pending |
| 4 | generator/perturb/twin/identifiability/complexity | **CHECKPOINT 3** — eyeball a small (5) generated set | ✅ built, tests pass | ⬜ pending |
| 5 | sampler/serialize | **CHECKPOINT 4** — inspect a serialized prompt payload | ✅ built, tests pass | ⬜ pending |
| 6 | harness + metered client + budget | **CHECKPOINT 5** — review budget estimate + transcript format; **approve cost ceiling** | ✅ built (mock only), tests pass | ⬜ pending |
| 7 | scoring | **CHECKPOINT 6** — scoring tests | ✅ built, tests pass | ⬜ pending |
| 8 | vertical slice end-to-end | **CHECKPOINT 7** — review full transcript + score (**"let it rip" gate**) | ✅ built (MockModel), smoke test passes | ⬜ pending |
| 9 | pilot + calibrate; corpora; full runs | **CHECKPOINT 8** — pre-registration freeze | ⛔ NOT started (gated) | ⛔ blocked on CP7 |

## Deliberate deviations from the spec (for reviewer attention)

1. **`Field` → `FieldSpec`.** The protocol's `Field` model is named `FieldSpec`
   in code to avoid colliding with `pydantic.Field` (used for field
   configuration). Purely a naming choice; the contract is otherwise identical.
2. **`cheap.provider` defaults to `mock`.** Out of the box the harness cannot
   make a real API call. Switching to a real provider is a deliberate edit
   plus an env-supplied API key — this operationalizes "no real calls until a
   human decides."
3. **Additive `RunOutcome` fields** (`queries_used`, `ood_row_accuracy`) beyond
   the spec's list — additive only, nothing removed.

## What a human must approve before Phase 7 ("let it rip")

1. CHECKPOINT 1: read the op semantics table (`docs/op_semantics.md`,
   generated from `ops_spec.py`) and confirm every edge case.
2. CHECKPOINT 5: approve the `usd_ceiling` and review a `budget.estimate`
   report for the intended grid.
3. CHECKPOINT 7: review the vertical-slice transcript + score, then authorize
   corpus generation and grid runs.
