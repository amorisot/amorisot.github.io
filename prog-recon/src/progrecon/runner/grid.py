"""Enumerate grid cells from config (Appendix C).

``enumerate_cells`` builds the cell list from ``config.yaml:grid``. It is
provided for Phase 7, but the naive full cross product is intentionally large —
the study runs targeted *sweeps*, not the full cross, and nothing here RUNS a
cell (running is gated by ``budget.guard`` + CHECKPOINT 7). ``single_cheap_cell``
builds the one cell the vertical slice uses.
"""

from __future__ import annotations

import itertools

from ..config import Config, get_config
from ..types import GridCell


def model_ids(cfg: Config) -> list[str]:
    ids = [cfg.models.cheap.id] + [m.id for m in cfg.models.roster]
    # de-dup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def enumerate_cells(cfg: Config | None = None) -> list[GridCell]:
    """The naive full cross product (large — for reference/Phase 7, not the slice)."""
    cfg = cfg or get_config()
    g = cfg.grid
    cells: list[GridCell] = []
    for n, m, k, corpus, band, noise, mode, qb, mid in itertools.product(
        g.n, g.m, g.k, g.corpus, g.complexity_band, g.noise, g.mode, g.query_budget, model_ids(cfg)
    ):
        cells.append(GridCell(
            n=n, m=m, k=k, corpus=corpus, complexity_band=band, noise=noise,  # type: ignore[arg-type]
            mode=mode, model_id=mid, query_budget=qb,  # type: ignore[arg-type]
        ))
    return cells


def single_cheap_cell(
    cfg: Config | None = None,
    *,
    n: int = 3,
    m: int = 1,
    k: int = 100,
    band: str = "med",
    noise: str = "none",
    mode: str = "scratch",
    query_budget: int = 0,
) -> GridCell:
    cfg = cfg or get_config()
    return GridCell(
        n=n, m=m, k=k, corpus="B", complexity_band=band, noise=noise,  # type: ignore[arg-type]
        mode=mode, model_id=cfg.models.cheap.id, query_budget=query_budget,  # type: ignore[arg-type]
    )
