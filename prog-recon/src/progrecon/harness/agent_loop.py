"""The HYPOTHESIZE -> TEST -> (QUERY) -> REFINE -> DONE agent loop.

Enforces the iteration budget ``B_iter``, query budget ``B_q``, and the per-run
token cap. TEST runs candidate programs in the locked-down ``sandbox`` against a
validation slice of TRAIN (never test). QUERY (only when ``cell.query_budget>0``)
hits the ground-truth evaluator for brand-new inputs. Emits a full transcript
and the run result.

D_TEST FIREWALL: the loop hard-asserts it was handed a TRAIN sample set and that
no test-split row ids are present, so test data can never reach the prompt.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..config import Config, get_config
from ..data import serialize
from ..dsl import executor
from ..match import row_match
from ..runner.model_client import ModelClient
from ..types import Example, GridCell, SampleSet, Task, Transform
from . import prompts, sandbox

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_QUERY_LINE = re.compile(r"^\s*QUERY:\s*(\[.*\])\s*$", re.MULTILINE)
# Row ids at/above this are test splits and must never reach the harness.
_TEST_ROW_FLOOR = 10_000_000


class LoopResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: str  # solved | failed | budget_exhausted | error
    final_source: str | None
    iterations_used: int
    queries_used: int
    tokens_used: int
    usd_cost: float
    transcript: list[dict[str, Any]]


def extract_code(text: str) -> str | None:
    matches = _CODE_BLOCK.findall(text)
    return matches[-1].strip() if matches else None


def extract_query(text: str) -> list[dict] | None:
    m = _QUERY_LINE.search(text)
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def _split_train(train: SampleSet, frac: float) -> tuple[SampleSet, SampleSet]:
    n = len(train.examples)
    n_val = max(1, int(round(n * frac))) if n > 1 else 0
    visible = train.examples[: n - n_val]
    validation = train.examples[n - n_val :] if n_val else train.examples
    return (
        SampleSet(kind="train", examples=visible or train.examples, seed=train.seed),
        SampleSet(kind="train", examples=validation, seed=train.seed),
    )


def _test_candidate(
    code: str, validation: list[Example], schema, *, rel_tol: float, timeout_s: float, mem_mb: int
) -> tuple[bool, list[str]]:
    results = sandbox.run_candidate_batch(
        code, [ex.x for ex in validation], timeout_s=timeout_s, mem_mb=mem_mb
    )
    failures: list[str] = []
    for ex, res in zip(validation, results, strict=True):
        ok = res.get("ok") and row_match(res.get("value"), ex.y, schema, rel_tol=rel_tol)
        if not ok:
            got = res.get("value") if res.get("ok") else f"<error: {res.get('error')}>"
            if len(failures) < 5:
                failures.append(f"  {ex.x} -> {got}  (expected {ex.y})")
    return (not failures, failures)


def run(
    task: Task,
    cell: GridCell,
    client: ModelClient,
    train: SampleSet,
    transform: Transform,
    *,
    config: Config | None = None,
    infill_source: str | None = None,
) -> LoopResult:
    cfg = config or get_config()

    # --- D_test firewall: the harness only ever sees TRAIN ---
    assert train.kind == "train", f"harness received a non-train split: {train.kind!r}"
    assert all(r < _TEST_ROW_FLOOR for r in train.row_ids), "test-split row ids leaked into the harness"

    mask_names = (cell.corpus in ("B", "twin")) or (not transform.semantic)
    visible, validation = _split_train(train, cfg.harness.train_validation_fraction)

    system = prompts.SYSTEM
    user0 = prompts.initial_user(
        visible, transform.schema, mask_names=mask_names, mode=cell.mode,
        infill_source=infill_source, query_budget=cell.query_budget,
    )
    messages: list[dict[str, Any]] = [{"role": "user", "content": user0}]
    transcript: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user0},
    ]

    final_source: str | None = None
    iterations = 0
    queries = 0
    tokens = 0
    start_usd = client.total_usd  # snapshot: report THIS run's spend, not cumulative
    outcome = "failed"

    while iterations < cfg.harness.b_iter:
        iterations += 1
        try:
            resp = client.complete(
                system=system, messages=messages, model_id=cell.model_id,
                max_tokens=cfg.harness.max_output_tokens,  # per-CALL output cap
            )
        except Exception as e:  # noqa: BLE001 - includes BudgetExceeded
            transcript.append({"role": "harness", "content": f"client error: {e}"})
            outcome = "error"
            break

        tokens += resp.input_tokens + resp.output_tokens
        messages.append({"role": "assistant", "content": resp.text})
        transcript.append({
            "role": "assistant", "content": resp.text,
            "tokens": resp.input_tokens + resp.output_tokens, "usd": resp.usd_cost,
        })

        if tokens > cfg.budget.per_run_token_cap:
            outcome = "budget_exhausted"
            break

        # QUERY turn (oracle on NEW inputs; never test data) ----------------
        query = extract_query(resp.text)
        if query is not None and queries < cell.query_budget:
            queries += 1
            rows = []
            for xi in query[:50]:
                try:
                    yi = executor.run_graph(transform.graph, xi)
                    rows.append(Example(row_id=-1, x=xi, y=yi))
                except Exception as e:  # noqa: BLE001
                    rows.append(Example(row_id=-1, x=xi, y={"error": str(e)}))
            qset = SampleSet(kind="train", examples=rows, seed=train.seed)
            results_text = serialize.for_prompt(qset, transform.schema, mask_names=mask_names)
            fb = prompts.query_results(results_text)
            messages.append({"role": "user", "content": fb})
            transcript.append({"role": "user", "content": fb})
            continue

        # HYPOTHESIZE/REFINE turn -------------------------------------------
        code = extract_code(resp.text)
        if code is None:
            fb = prompts.no_code_feedback()
            messages.append({"role": "user", "content": fb})
            transcript.append({"role": "user", "content": fb})
            continue

        final_source = code
        passed, failures = _test_candidate(
            code, validation.examples, transform.schema,
            rel_tol=cfg.scoring.rel_tol, timeout_s=cfg.sandbox.timeout_s, mem_mb=cfg.sandbox.mem_mb,
        )
        if passed:
            outcome = "solved"
            transcript.append({"role": "harness", "content": "TEST passed on train validation slice"})
            break
        fb = prompts.test_feedback(len(failures), len(validation.examples), failures)
        messages.append({"role": "user", "content": fb})
        transcript.append({"role": "user", "content": fb})

    if outcome == "failed" and iterations >= cfg.harness.b_iter:
        outcome = "budget_exhausted"

    return LoopResult(
        outcome=outcome, final_source=final_source, iterations_used=iterations,
        queries_used=queries, tokens_used=tokens, usd_cost=client.total_usd - start_usd, transcript=transcript,
    )


# Re-export for convenience / firewall tests.
TEST_ROW_FLOOR = _TEST_ROW_FLOOR
__all__ = ["run", "LoopResult", "extract_code", "extract_query", "TEST_ROW_FLOOR"]
