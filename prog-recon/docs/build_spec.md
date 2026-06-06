# Build Specification — Program Reconstruction Scaling Study

This file preserves the engineering build specification that this repository
implements. The companion scientific protocol
(`program_reconstruction_protocol.md`) governs *intent*; this spec governs
*implementation*. Where they disagree, the protocol governs intent and this doc
governs implementation.

**Golden rule:** build in phases, in order. Stop at every `# CHECKPOINT` and
wait for human approval before continuing. Do not generate the full corpora or
run the full grid until Phase 7.

## 0. Non-negotiable constraints

1. No real model API calls until Phase 6, and even then only against the
   configured "cheap" model on a single grid cell. The full grid (Phase 7)
   requires explicit human sign-off on the cost estimate produced in Phase 5.
2. Every API call goes through one rate-limited, cost-metered client
   (`runner/model_client.py`). No module calls a provider SDK directly. The
   client enforces a hard global spend ceiling and aborts if exceeded.
3. Ground-truth correctness is sacred. The DSL executor and every op have unit
   tests with hand-checked expected values (Phase 2). Do not proceed past
   Phase 2 without 100% of op tests passing and human review of the op
   semantics table.
4. Determinism & seeds. Every random choice is driven by an explicit seed
   recorded in the task/run manifest. Re-running reproduces the same task and
   the same `D_test`.
5. The agent under test never sees `D_test`. Enforced in code: separate
   objects, separate file paths, an assertion in the scorer that `D_test` ids
   are disjoint from anything the harness exposed.
6. No network access in the executor sandbox. Candidate programs and
   ground-truth programs run in a restricted subprocess: no network, no
   filesystem writes outside a temp dir, wall-clock + memory cap.

## 1. Tech stack

Python 3.11+, `uv` (fallback pip), `ruff` + `black`, `pydantic` v2 for all data
contracts, `mypy` clean, `pytest`, `pydantic-settings` reading a single
`config.yaml` + env for secrets. No heavyweight framework.

(See the repo `README.md` for the directory layout, which mirrors the spec.)

## Phases & checkpoints

| Phase | Output | Gate |
|---|---|---|
| 1 | `types.py` | CHECKPOINT 0 |
| 2 | DSL ops + semantics table | CHECKPOINT 1 (most critical) |
| 3 | executor + sandbox | CHECKPOINT 2 |
| 4 | generator/perturb/twin/identifiability/complexity | CHECKPOINT 3 |
| 5 | sampler/serialize | CHECKPOINT 4 |
| 6 | harness + metered client + budget | CHECKPOINT 5 |
| 7 | scoring | CHECKPOINT 6 |
| 8 | vertical slice end-to-end | CHECKPOINT 7 ("let it rip" gate) |
| 9 | pilot + calibrate; corpora; full runs | CHECKPOINT 8 (pre-reg freeze) |

## 13. What the agent must NOT do

- Do not call any provider SDK outside `model_client`.
- Do not generate all 100 transformations or enumerate the full grid before
  Checkpoint 7.
- Do not invent op edge-case semantics silently — add a row to the semantics
  table and surface the decision.
- Do not let any `test_*` data reach the harness or the prompt.
- Do not maintain `Transform.source` and `Transform.graph` independently —
  codegen source from graph.
- Do not proceed past a CHECKPOINT without explicit human approval.
- Do not raise the `usd_ceiling` or skip `budget.estimate` to "save time."

> The op semantics table is generated as `docs/op_semantics.md` from
> `src/progrecon/dsl/ops_spec.py` (the single source of truth). Regenerate it
> with `uv run progrecon dump-ops`.
