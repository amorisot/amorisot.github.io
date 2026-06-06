"""Targeted, resumable experiment sweeps (build spec §10.5).

The study runs *sweeps* (one experimental lever varied at a time), NOT the naive
full cross product. Each sweep expands to a list of ``WorkItem`` (a concrete
transform + a grid cell + a repeat index + a perturbation). ``run_sweep`` drives
them through ``run_cell``, and is **resumable**: a work item whose per-run
manifest already exists is loaded and skipped, so an interrupted sweep continues
where it left off.

``estimate`` (re-exported from ``budget``) is computed from the plan's cells,
so you can see and approve the spend before anything runs.

Offline by default: when no client is supplied, ``run_cell`` builds a per-task
oracle ``MockModel`` (proves the plumbing, firewall, scoring, and resumability
without a network). For real runs, pass one shared metered ``ModelClient`` so
the global USD ceiling is enforced across the whole sweep.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..config import Config, get_config
from ..corpus.complexity import transform_band
from ..runner.model_client import ModelClient
from ..types import GridCell, PerturbationSpec, RunOutcome, Transform
from . import budget, run_cell


def _seed(*parts: object) -> int:
    """Deterministic (cross-process) seed from parts — keeps runs resumable."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


@dataclass
class WorkItem:
    transform: Transform
    cell: GridCell
    repeat_idx: int
    perturbation: PerturbationSpec
    task_id: str


def _item(
    t: Transform, corpus_label: str, *, k: int, model_id: str, repeat: int,
    noise: str = "none", mode: str = "scratch", query_budget: int = 0, n: int | None = None,
) -> WorkItem:
    n_eff = t.schema.n if n is None else n
    distractors = max(0, n_eff - t.schema.n)
    seed = _seed(t.id, corpus_label, k, n_eff, query_budget, noise, mode, repeat)
    cell = GridCell(
        n=n_eff, m=t.schema.m, k=k, corpus=corpus_label,  # type: ignore[arg-type]
        complexity_band=transform_band(t), noise=noise,  # type: ignore[arg-type]
        mode=mode, model_id=model_id, query_budget=query_budget,  # type: ignore[arg-type]
    )
    spec = PerturbationSpec(seed=seed, n_distractors=distractors)
    return WorkItem(transform=t, cell=cell, repeat_idx=repeat, perturbation=spec, task_id=f"{t.id}#p{seed}")


# ---------------------------------------------------------------------------
# Sweep planners (one lever each)
# ---------------------------------------------------------------------------


def plan_sample_sweep(
    transforms: list[Transform], corpus_label: str, *, ks: list[int], repeats: int, model_id: str
) -> list[WorkItem]:
    """Claim 2: vary k (sample size) at fixed everything else."""
    return [
        _item(t, corpus_label, k=k, model_id=model_id, repeat=r)
        for t in transforms for k in ks for r in range(repeats)
    ]


def plan_dimensionality_sweep(
    transforms: list[Transform], corpus_label: str, *, distractor_levels: list[int],
    k: int, repeats: int, model_id: str
) -> list[WorkItem]:
    """Claim 1: vary n via irrelevant distractor inputs (the clean n-lever)."""
    return [
        _item(t, corpus_label, k=k, model_id=model_id, repeat=r, n=t.schema.n + d)
        for t in transforms for d in distractor_levels for r in range(repeats)
    ]


def plan_semantic_sweep(
    pairs: list[tuple[Transform, Transform]], *, k: int, repeats: int, model_id: str
) -> list[WorkItem]:
    """Claim 3: matched semantic (A) vs abstract (twin) at identical structure."""
    items: list[WorkItem] = []
    for a, twin in pairs:
        for r in range(repeats):
            items.append(_item(a, "A", k=k, model_id=model_id, repeat=r))
            items.append(_item(twin, "twin", k=k, model_id=model_id, repeat=r))
    return items


def plan_leakage_sweep(
    transforms: list[Transform], corpus_label: str, *, query_budgets: list[int],
    k: int, repeats: int, model_id: str
) -> list[WorkItem]:
    """Claim 4: vary the oracle query budget B_q."""
    return [
        _item(t, corpus_label, k=k, model_id=model_id, repeat=r, query_budget=qb)
        for t in transforms for qb in query_budgets for r in range(repeats)
    ]


# ---------------------------------------------------------------------------
# Estimation + execution
# ---------------------------------------------------------------------------


def estimate(items: list[WorkItem], *, cfg: Config | None = None) -> budget.BudgetReport:
    cfg = cfg or get_config()
    return budget.estimate([wi.cell for wi in items], repeats=1, cfg=cfg)


def run_sweep(
    items: list[WorkItem],
    *,
    client: ModelClient | None = None,
    cfg: Config | None = None,
    resume: bool = True,
) -> list[RunOutcome]:
    """Run all work items (resumable). Returns the per-item RunOutcomes.

    If ``client`` is given (a real metered client), it is shared across items so
    the global USD ceiling applies to the whole sweep. If ``None``, each item
    uses a per-task oracle MockModel (offline).
    """
    cfg = cfg or get_config()
    outcomes: list[RunOutcome] = []
    for wi in items:
        path = run_cell.manifest_path(wi.task_id, wi.cell, wi.repeat_idx, cfg)
        if resume and path.exists():
            outcomes.append(RunOutcome.model_validate_json(path.read_text()))
            continue
        res = run_cell.run_cell(
            wi.cell, transform=wi.transform, client=client, repeat_idx=wi.repeat_idx,
            seed=wi.perturbation.seed, perturbation=wi.perturbation, cfg=cfg, write_artifacts=True,
        )
        outcomes.append(res.outcome)
    return outcomes
