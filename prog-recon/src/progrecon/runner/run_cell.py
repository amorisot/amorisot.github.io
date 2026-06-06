"""Run one grid cell end-to-end (the vertical slice spine).

Generate (or accept) one identified transform -> build train/test_id/test_ood
with the D_test firewall -> run the harness with a model client -> score
against both test splits -> emit a RunOutcome + transcript.

For offline/CI use, when the configured provider is ``mock`` and no client is
supplied, an *oracle* MockModel that returns the ground-truth program is used.
This proves the full pipeline (loop, firewall, scoring, cost logging) without a
network. With a real provider configured, the real metered client is used.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ..config import Config, get_config
from ..corpus import complexity, generator, identifiability, perturb
from ..data import sampler
from ..harness import agent_loop
from ..runner.model_client import MockModel, ModelClient
from ..scoring import metrics
from ..types import GridCell, PerturbationSpec, RunOutcome, Transform

_DEPTH_FOR_BAND = {"low": 2, "med": 3, "high": 4}


class CellRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome: RunOutcome
    transcript_path: str
    transform_id: str
    complexity_c: float
    one_line: str


def generate_identified_transform(
    cell: GridCell, *, seed: int, cfg: Config | None = None
) -> Transform:
    """Generate a Corpus-B transform for the cell that passes identifiability."""
    cfg = cfg or get_config()
    persona = cfg.generators[min(len(cfg.generators) - 1, 1)]
    depth = _DEPTH_FOR_BAND.get(cell.complexity_band, 3)
    for attempt in range(60):
        t = generator.sample_transform(
            persona, n=cell.n, m=cell.m, target_depth=depth,
            seed=seed + attempt * 104729, transform_id=f"B-CELL-{cell.cell_key()}-{seed}",
        )
        rep = identifiability.check(t, n_probes=300)
        if rep.passed:
            return t
    raise RuntimeError(f"could not generate an identified transform for cell {cell.cell_key()}")


def _oracle_client(transform: Transform, cfg: Config) -> ModelClient:
    src = transform.source

    def responder(system, messages):  # noqa: ANN001 - signature fixed by MockModel
        return f"```python\n{src}\n```\nDONE"

    return ModelClient(cfg, mock_model=MockModel(responder=responder))


def manifest_path(task_id: str, cell: GridCell, repeat_idx: int, cfg: Config) -> Path:
    """Deterministic per-run manifest path (used for writing AND resumability)."""
    return Path(cfg.paths.manifests_dir) / f"{task_id}__{cell.cell_key()}__r{repeat_idx}.json"


def run_cell(
    cell: GridCell,
    *,
    transform: Transform | None = None,
    client: ModelClient | None = None,
    repeat_idx: int = 0,
    seed: int = 0,
    perturbation: PerturbationSpec | None = None,
    cfg: Config | None = None,
    write_artifacts: bool = True,
) -> CellRun:
    cfg = cfg or get_config()
    if transform is None:
        transform = generate_identified_transform(cell, seed=seed, cfg=cfg)

    spec = perturbation or PerturbationSpec(seed=seed)
    task = perturb.make_task(transform, spec)
    concrete = task.transform

    bundle = sampler.make_dataset(
        concrete, k=cell.k, seed=seed, branch_bias=True, noise=cell.noise, cfg=cfg
    )
    assert bundle.firewall_ok(), "D_test firewall breach in run_cell"

    if client is None:
        provider = cfg.models.cheap.provider
        if provider == "mock" or cell.model_id == "mock":
            client = _oracle_client(concrete, cfg)
        else:  # pragma: no cover - real provider path, never exercised offline
            client = ModelClient(cfg)

    loop = agent_loop.run(task, cell, client, bundle.train, concrete, config=cfg)

    transcript_path = ""
    if write_artifacts:
        logs_dir = Path(cfg.paths.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = str(logs_dir / f"{task.task_id}__{cell.cell_key()}__r{repeat_idx}.json")
        Path(transcript_path).write_text(json.dumps(loop.transcript, indent=2, default=str))

    outcome = RunOutcome(
        task_id=task.task_id, cell=cell, repeat_idx=repeat_idx,
        final_source=loop.final_source, outcome=loop.outcome,
        iterations_used=loop.iterations_used, queries_used=loop.queries_used,
        tokens_used=loop.tokens_used, usd_cost=loop.usd_cost,
        transcript_path=transcript_path or "(not written)",
    )
    outcome = metrics.score_run(outcome, concrete, bundle.test_id, bundle.test_ood, cfg=cfg)

    if write_artifacts:
        man_dir = Path(cfg.paths.manifests_dir)
        man_dir.mkdir(parents=True, exist_ok=True)
        manifest_path(task.task_id, cell, repeat_idx, cfg).write_text(outcome.model_dump_json(indent=2))

    c_val = complexity.transform_complexity(concrete)
    one_line = (
        f"[{cell.cell_key()}] {outcome.outcome} "
        f"exact_id={outcome.exact_pass} exact_ood={outcome.ood_exact_pass} "
        f"row_acc={outcome.row_accuracy:.3f} iters={outcome.iterations_used} "
        f"tokens={outcome.tokens_used} cost=${outcome.usd_cost:.6f} C={c_val:.1f}"
    )
    return CellRun(
        outcome=outcome, transcript_path=transcript_path or "(not written)",
        transform_id=concrete.id, complexity_c=c_val, one_line=one_line,
    )
