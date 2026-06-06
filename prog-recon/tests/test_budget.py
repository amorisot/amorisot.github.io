"""Budget estimator + ceiling refusal tests (CHECKPOINT 5)."""

import pytest

from progrecon.config import load_config
from progrecon.runner import budget
from progrecon.types import GridCell

CFG = load_config()


def _cell(model_id):
    return GridCell(
        n=3, m=1, k=100, corpus="B", complexity_band="med", noise="none",
        mode="scratch", model_id=model_id, query_budget=0,
    )


def test_estimate_sums_costs():
    cells = [_cell(CFG.models.cheap.id), _cell("claude-sonnet-4-6")]
    rep = budget.estimate(cells, repeats=2, cfg=CFG)
    assert rep.n_cells == 2 and rep.repeats == 2
    assert rep.total_usd > 0
    assert rep.total_usd == pytest.approx(sum(c.usd for c in rep.per_cell))
    assert "Budget estimate" in rep.render()


def test_guard_refuses_over_ceiling_without_accept():
    # Tiny ceiling forces the estimate over budget.
    cfg = CFG.model_copy(update={"budget": CFG.budget.model_copy(update={"usd_ceiling": 1e-9})})
    rep = budget.estimate([_cell("claude-opus-4-8")], cfg=cfg)
    assert not rep.within_ceiling
    with pytest.raises(budget.BudgetRefused):
        budget.guard(rep, accept_cost=False)
    # explicit acceptance lets it through
    budget.guard(rep, accept_cost=True)


def test_guard_passes_within_ceiling():
    rep = budget.estimate([_cell(CFG.models.cheap.id)], cfg=CFG)
    assert rep.within_ceiling
    budget.guard(rep, accept_cost=False)  # no raise
