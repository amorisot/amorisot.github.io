"""Primary + secondary metrics (build spec §6.3).

* ``score_run`` fills a RunOutcome's scoring fields by evaluating its final
  program against the in-distribution and OOD test splits (separately).
* ``aggregate_cell`` rolls a set of scored RunOutcomes up to cell level:
  exact-reconstruction rate (primary), OOD exact rate, budget-exhaustion rate,
  mean iterations/tokens-to-solve, and mean partial accuracies.
* ``ast_similarity`` is computed but flagged EXPLORATORY — it never gates
  success.
"""

from __future__ import annotations

import ast
import difflib

from pydantic import BaseModel, ConfigDict

from ..config import Config, get_config
from ..types import RunOutcome, SampleSet, Transform
from . import equivalence


class CellMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_runs: int
    exact_reconstruction_rate: float  # PRIMARY: fraction with exact_pass on test_id
    ood_exact_rate: float
    budget_exhaustion_rate: float
    error_rate: float
    mean_row_accuracy: float
    mean_field_accuracy: float
    mean_ood_row_accuracy: float
    mean_iterations_to_solve: float | None
    mean_tokens_to_solve: float | None


def score_run(
    outcome: RunOutcome,
    transform: Transform,
    test_id: SampleSet,
    test_ood: SampleSet,
    *,
    cfg: Config | None = None,
) -> RunOutcome:
    """Return a copy of ``outcome`` with scoring fields filled from the test splits."""
    cfg = cfg or get_config()
    if not outcome.final_source:
        return outcome.model_copy(update={
            "exact_pass": False, "row_accuracy": 0.0, "field_accuracy": 0.0,
            "ood_exact_pass": False, "ood_row_accuracy": 0.0,
        })
    rid = equivalence.passes(outcome.final_source, transform, test_id, cfg=cfg)
    rood = equivalence.passes(outcome.final_source, transform, test_ood, cfg=cfg)
    return outcome.model_copy(update={
        "exact_pass": rid.exact_pass,
        "row_accuracy": rid.row_accuracy,
        "field_accuracy": rid.field_accuracy,
        "ood_exact_pass": rood.exact_pass,
        "ood_row_accuracy": rood.row_accuracy,
    })


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def aggregate_cell(outcomes: list[RunOutcome]) -> CellMetrics:
    n = len(outcomes)
    if n == 0:
        return CellMetrics(
            n_runs=0, exact_reconstruction_rate=0.0, ood_exact_rate=0.0,
            budget_exhaustion_rate=0.0, error_rate=0.0, mean_row_accuracy=0.0,
            mean_field_accuracy=0.0, mean_ood_row_accuracy=0.0,
            mean_iterations_to_solve=None, mean_tokens_to_solve=None,
        )
    exact = sum(1 for o in outcomes if o.exact_pass)
    ood = sum(1 for o in outcomes if o.ood_exact_pass)
    exhausted = sum(1 for o in outcomes if o.outcome == "budget_exhausted")
    errored = sum(1 for o in outcomes if o.outcome == "error")
    row_accs = [o.row_accuracy for o in outcomes if o.row_accuracy is not None]
    field_accs = [o.field_accuracy for o in outcomes if o.field_accuracy is not None]
    ood_accs = [o.ood_row_accuracy for o in outcomes if o.ood_row_accuracy is not None]
    solved = [o for o in outcomes if o.outcome == "solved"]
    return CellMetrics(
        n_runs=n,
        exact_reconstruction_rate=exact / n,
        ood_exact_rate=ood / n,
        budget_exhaustion_rate=exhausted / n,
        error_rate=errored / n,
        mean_row_accuracy=(_mean(row_accs) or 0.0),
        mean_field_accuracy=(_mean(field_accs) or 0.0),
        mean_ood_row_accuracy=(_mean(ood_accs) or 0.0),
        mean_iterations_to_solve=_mean([float(o.iterations_used) for o in solved]),
        mean_tokens_to_solve=_mean([float(o.tokens_used) for o in solved]),
    )


def ast_similarity(candidate_source: str, ground_truth_source: str) -> float:
    """EXPLORATORY ONLY — structural similarity in [0,1]; never gates success.

    Compares normalized AST dumps via a sequence ratio. Falls back to 0.0 if the
    candidate doesn't parse.
    """
    try:
        a = ast.dump(ast.parse(candidate_source))
        b = ast.dump(ast.parse(ground_truth_source))
    except SyntaxError:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()
