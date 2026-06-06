"""Cost estimation + hard ceiling enforcement (build spec §Module budget).

``estimate`` computes expected calls × avg tokens × price for a set of grid
cells and produces a report. ``guard`` refuses to proceed when the estimate is
over the configured ceiling unless the caller passes an explicit acceptance
flag (the CLI's ``--i-accept-cost``). This is the gate that stands between the
vertical slice and the full grid.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..config import Config, get_config
from ..types import GridCell


class CellEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cell_key: str
    model_id: str
    calls: int
    input_tokens: int
    output_tokens: int
    usd: float


class BudgetReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_usd: float
    ceiling_usd: float
    within_ceiling: bool
    n_cells: int
    repeats: int
    per_cell: list[CellEstimate]

    def render(self) -> str:
        lines = [
            "Budget estimate",
            "===============",
            f"cells={self.n_cells}  repeats={self.repeats}  "
            f"estimated total=${self.total_usd:.2f}  ceiling=${self.ceiling_usd:.2f}  "
            f"within_ceiling={self.within_ceiling}",
            "",
        ]
        for c in sorted(self.per_cell, key=lambda e: e.usd, reverse=True)[:25]:
            lines.append(
                f"  {c.cell_key:<48} {c.model_id:<28} calls={c.calls:<5} ${c.usd:.3f}"
            )
        if len(self.per_cell) > 25:
            lines.append(f"  ... ({len(self.per_cell) - 25} more cells)")
        return "\n".join(lines)


def estimate(
    cells: list[GridCell],
    *,
    repeats: int = 1,
    avg_input_tokens: int = 4000,
    avg_output_tokens: int = 600,
    cfg: Config | None = None,
) -> BudgetReport:
    """Estimate spend for running ``cells`` × ``repeats``.

    Per task, the harness makes up to ``b_iter`` model calls; we estimate the
    expected number of calls as ``b_iter`` (worst-case upper bound is honest for
    a *ceiling* estimate).
    """
    cfg = cfg or get_config()
    calls_per_task = cfg.harness.b_iter
    per_cell: list[CellEstimate] = []
    total = 0.0
    for cell in cells:
        price = cfg.price_for(cell.model_id)
        calls = calls_per_task * repeats
        in_tok = calls * avg_input_tokens
        out_tok = calls * avg_output_tokens
        usd = in_tok / 1_000_000 * price.input_per_mtok + out_tok / 1_000_000 * price.output_per_mtok
        total += usd
        per_cell.append(
            CellEstimate(
                cell_key=cell.cell_key(), model_id=cell.model_id, calls=calls,
                input_tokens=in_tok, output_tokens=out_tok, usd=usd,
            )
        )
    ceiling = cfg.budget.usd_ceiling
    return BudgetReport(
        total_usd=total, ceiling_usd=ceiling, within_ceiling=total <= ceiling,
        n_cells=len(cells), repeats=repeats, per_cell=per_cell,
    )


class BudgetRefused(Exception):
    pass


def guard(report: BudgetReport, *, accept_cost: bool) -> None:
    """Refuse to run when the estimate is over the ceiling without explicit accept."""
    if report.within_ceiling:
        return
    if not accept_cost:
        raise BudgetRefused(
            f"estimated ${report.total_usd:.2f} exceeds ceiling ${report.ceiling_usd:.2f}; "
            "re-run with --i-accept-cost to proceed deliberately (or lower the grid / raise the "
            "ceiling in config.yaml)"
        )
