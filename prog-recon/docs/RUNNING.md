# Running the experiments

Everything to date runs **offline** with a `MockModel`. To run the real sweeps
against a model you need five things, in order.

## 1. Network egress to the provider
This managed environment's outbound network is governed by its policy. WebFetch
to external hosts is currently **blocked**, which means `api.anthropic.com` is
likely unreachable too. You need an environment/network policy that permits the
provider host (see https://code.claude.com/docs/en/claude-code-on-the-web).

## 2. An API key (env only — never in config)
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```
The client reads secrets from the environment only.

## 3. Point the cheap model at a real provider
In `config.yaml`, flip the provider (it defaults to `mock` so nothing hits a
network by accident):
```yaml
models:
  cheap: {id: claude-haiku-4-5-20251001, provider: anthropic, version: "2026-01", temperature: 0}
  roster: []   # add more models here for cross-model sweeps
```
Pricing for the roster models is already in `config.yaml:model_pricing`.

## 4. Approve the cost
```bash
uv run progrecon build-corpus            # cache the corpus once (offline, free)
uv run progrecon sweep --name sample --repeats 3   # prints the $ estimate, runs nothing
```
The estimate is computed from the plan (runs × B_iter × avg tokens × price) and
is gated by `budget.usd_ceiling`. To execute, pass `--run` (and, only if the
estimate is over the ceiling, `--i-accept-cost`). The metered client aborts
mid-sweep with `BudgetExceeded` if cumulative spend would cross the ceiling.

Indicative estimates today (Haiku, 5 transforms × 3 repeats, $50 ceiling):
`sample` 60 runs ≈ $4.20 · `dimensionality` 60 ≈ $4.20 · `leakage` 45 ≈ $3.15 ·
`semantic` 30 ≈ $2.10. Scale by transforms/repeats/models for the full study.

## 5. Decisions still needed from a human (the protocol's gates)
- **Pilot first** (protocol §3.9): a few transforms × 2 models × a few cells to
  calibrate the `C(T)` weights, then write them back to `config.yaml`. Discard
  pilot data from the final set.
- **Pre-registration freeze** (CHECKPOINT 8): tag the repo + protocol version
  *between* the pilot and the test-split runs.
- **Roster + sweep design**: which models; which sweeps/repeats; `k` grid;
  query budgets; `test_id`/`test_ood` sizes (2000 each is heavy × many cells —
  consider lowering for the pilot).

## Running, resuming, analyzing
```bash
uv run progrecon sweep --name sample --repeats 3 --run [--i-accept-cost]
uv run progrecon analyze        # tidy per-cell table from the run manifests
```
Sweeps are **resumable**: each run writes a manifest keyed by
`task × cell × repeat`; re-running skips completed work, so an interrupted or
budget-aborted sweep continues where it left off.
