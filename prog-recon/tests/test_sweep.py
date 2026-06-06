"""Sweep planning + resumable execution tests (offline, MockModel oracle)."""

import pytest

from progrecon.config import load_config
from progrecon.corpus import authored, twin
from progrecon.runner import run_cell, sweep


@pytest.fixture
def cfg(tmp_path):
    c = load_config()
    return c.model_copy(update={
        "scoring": c.scoring.model_copy(update={"test_id_size": 50, "test_ood_size": 50}),
        "paths": c.paths.model_copy(update={
            "manifests_dir": str(tmp_path / "manifests"), "logs_dir": str(tmp_path / "logs"),
        }),
    })


@pytest.fixture
def transforms():
    return authored.build_all()[:2]


def test_sample_plan_size_and_cells(transforms, cfg):
    items = sweep.plan_sample_sweep(transforms, "B", ks=[10, 100], repeats=2, model_id="mock")
    assert len(items) == 2 * 2 * 2  # transforms * ks * repeats
    assert {wi.cell.k for wi in items} == {10, 100}
    assert all(wi.cell.corpus == "B" and wi.cell.model_id == "mock" for wi in items)
    # deterministic seeds -> stable task ids (resumability depends on this)
    again = sweep.plan_sample_sweep(transforms, "B", ks=[10, 100], repeats=2, model_id="mock")
    assert [wi.task_id for wi in items] == [wi.task_id for wi in again]


def test_dimensionality_plan_adds_distractors(transforms, cfg):
    items = sweep.plan_dimensionality_sweep(transforms, "B", distractor_levels=[0, 5], k=20, repeats=1, model_id="mock")
    base = transforms[0]
    base_items = [wi for wi in items if wi.transform is base]
    ns = sorted(wi.cell.n for wi in base_items)
    assert ns == [base.schema.n, base.schema.n + 5]
    # the n-lever is implemented via distractor inputs
    assert any(wi.perturbation.n_distractors == 5 for wi in items)


def test_semantic_plan_pairs_a_with_twin(transforms, cfg):
    a = transforms[0]
    tw = twin.make_abstract(a, seed=1)
    items = sweep.plan_semantic_sweep([(a, tw)], k=20, repeats=1, model_id="mock")
    assert len(items) == 2
    assert {wi.cell.corpus for wi in items} == {"A", "twin"}
    assert {wi.cell.k for wi in items} == {20}


def test_estimate_is_computed_from_plan(transforms, cfg):
    items = sweep.plan_sample_sweep(transforms, "B", ks=[10, 100], repeats=1, model_id=cfg.models.cheap.id)
    rep = sweep.estimate(items, cfg=cfg)
    assert rep.total_usd > 0
    assert rep.within_ceiling


def test_run_sweep_offline_and_resumable(transforms, cfg, monkeypatch):
    items = sweep.plan_sample_sweep(transforms[:1], "B", ks=[10, 30], repeats=1, model_id="mock")
    outcomes = sweep.run_sweep(items, cfg=cfg)
    assert len(outcomes) == 2
    # oracle MockModel returns ground truth -> solved + exact in-distribution
    assert all(o.outcome == "solved" and o.exact_pass for o in outcomes)

    # Resumability: every manifest now exists, so a second pass must NOT re-run.
    def _boom(*a, **k):  # pragma: no cover - asserts it is never called
        raise AssertionError("run_cell should have been skipped on resume")

    monkeypatch.setattr(run_cell, "run_cell", _boom)
    again = sweep.run_sweep(items, cfg=cfg, resume=True)
    assert [o.task_id for o in again] == [o.task_id for o in outcomes]
    assert all(o.exact_pass for o in again)
