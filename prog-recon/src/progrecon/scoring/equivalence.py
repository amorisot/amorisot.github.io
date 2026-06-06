"""Behavioral-equivalence predicate (build spec §6.2).

Runs a candidate program in the sandbox over a TEST split and compares to ground
truth: numeric within ``rel_tol``, categorical/bool exact; a row passes iff ALL
output fields match; ``exact_pass`` iff every row passes. Also reports
``row_accuracy``, ``field_accuracy``, and per-field accuracy.

HARD ASSERTION: the supplied sample set must be a TEST split, never train.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..config import Config, get_config
from ..harness.sandbox import run_candidate_batch
from ..match import row_match, value_match
from ..types import SampleSet, Transform


class EquivResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    exact_pass: bool
    n_rows: int
    n_pass: int
    row_accuracy: float
    field_accuracy: float
    per_field_accuracy: dict[str, float]
    n_errors: int  # rows where the candidate raised / crashed / wasn't serializable


def passes(
    candidate_source: str,
    transform: Transform,
    test_set: SampleSet,
    *,
    cfg: Config | None = None,
    rel_tol: float | None = None,
    timeout_s: float | None = None,
    mem_mb: int | None = None,
) -> EquivResult:
    assert test_set.is_test, (
        f"equivalence.passes was given a {test_set.kind!r} split — scoring runs on TEST only"
    )
    cfg = cfg or get_config()
    rel_tol = cfg.scoring.rel_tol if rel_tol is None else rel_tol
    # Give the batch generous wall-clock headroom for large test splits.
    timeout_s = max(cfg.sandbox.timeout_s, 30.0) if timeout_s is None else timeout_s
    mem_mb = cfg.sandbox.mem_mb if mem_mb is None else mem_mb

    schema = transform.schema
    out_fields = [f.name for f in schema.outputs]
    xs = [ex.x for ex in test_set.examples]
    results = run_candidate_batch(candidate_source, xs, timeout_s=timeout_s, mem_mb=mem_mb)

    n = len(test_set.examples)
    n_pass = 0
    n_errors = 0
    per_field_hits = {name: 0 for name in out_fields}
    for ex, res in zip(test_set.examples, results, strict=True):
        if not res.get("ok"):
            n_errors += 1
            continue
        y_pred = res.get("value")
        if row_match(y_pred, ex.y, schema, rel_tol=rel_tol):
            n_pass += 1
        if isinstance(y_pred, dict):
            for name in out_fields:
                if name in y_pred and value_match(y_pred[name], ex.y[name], rel_tol=rel_tol):
                    per_field_hits[name] += 1

    field_total = n * len(out_fields) if out_fields else 1
    field_hits = sum(per_field_hits.values())
    return EquivResult(
        kind=test_set.kind,
        exact_pass=(n_pass == n and n > 0),
        n_rows=n,
        n_pass=n_pass,
        row_accuracy=(n_pass / n if n else 0.0),
        field_accuracy=(field_hits / field_total if field_total else 0.0),
        per_field_accuracy={k: (v / n if n else 0.0) for k, v in per_field_hits.items()},
        n_errors=n_errors,
    )
