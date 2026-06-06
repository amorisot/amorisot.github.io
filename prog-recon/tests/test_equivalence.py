"""Scoring tests (CHECKPOINT 6)."""

import pytest

from progrecon.data import sampler
from progrecon.scoring import equivalence, metrics
from progrecon.types import DataflowGraph, FieldSpec, GridCell, OpNode, RunOutcome, Schema

from .conftest import build_transform


def _mod_transform():
    schema = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=30)],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="mod_k", params={"k": 4}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    return build_transform(schema, g, tid="B-score")


def _identity_transform():
    schema = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=10)],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="affine", params={"a": 1, "b": 0}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    return build_transform(schema, g, tid="B-id")


def test_correct_candidate_passes_exact():
    t = _mod_transform()
    test = sampler.make_samples(t, 200, 1, "test_id")
    r = equivalence.passes(t.source, t, test)
    assert r.exact_pass and r.row_accuracy == 1.0 and r.n_errors == 0


def test_off_by_some_fails_exact_but_partial_accuracy():
    t = _mod_transform()
    test = sampler.make_samples(t, 200, 2, "test_id")
    # Correct except it adds 1 on even inputs -> some rows wrong.
    cand = "def transform(x):\n    v = x['in_0']\n    return {'out_0': (v % 4) + (1 if v % 2 == 0 else 0)}"
    r = equivalence.passes(cand, t, test)
    assert not r.exact_pass
    assert 0.0 < r.row_accuracy < 1.0


def test_overfit_to_train_fails_ood():
    t = _identity_transform()  # out_0 == in_0, training range [0,10]
    test_id = sampler.make_samples(t, 300, 5, "test_id")
    test_ood = sampler.make_samples(t, 300, 5, "test_ood")
    # Candidate matches identity only inside [0,10]; returns 0 outside.
    cand = "def transform(x):\n    v = x['in_0']\n    return {'out_0': v if 0 <= v <= 10 else 0}"
    rid = equivalence.passes(cand, t, test_id)
    rood = equivalence.passes(cand, t, test_ood)
    assert rid.exact_pass            # perfect in-distribution
    assert not rood.exact_pass       # but wrong out-of-distribution
    assert rood.row_accuracy < 1.0


def test_firewall_rejects_train_split():
    t = _mod_transform()
    train = sampler.make_samples(t, 50, 1, "train")
    with pytest.raises(AssertionError):
        equivalence.passes(t.source, t, train)


def test_candidate_error_counts_as_fail():
    t = _mod_transform()
    test = sampler.make_samples(t, 50, 1, "test_id")
    r = equivalence.passes("def transform(x):\n    return {'out_0': 1/0}", t, test)
    assert not r.exact_pass and r.n_errors == len(test.examples)


# --- metrics ---------------------------------------------------------------


def _outcome(outcome, exact, ood_exact, row_acc, iters, tokens):
    cell = GridCell(n=1, m=1, k=10, corpus="B", complexity_band="low", noise="none",
                    mode="scratch", model_id="mock", query_budget=0)
    return RunOutcome(
        task_id="t", cell=cell, repeat_idx=0, final_source="x", outcome=outcome,
        iterations_used=iters, tokens_used=tokens, usd_cost=0.0, transcript_path="x.json",
        exact_pass=exact, row_accuracy=row_acc, field_accuracy=row_acc,
        ood_exact_pass=ood_exact, ood_row_accuracy=(1.0 if ood_exact else 0.5),
    )


def test_aggregate_cell():
    outs = [
        _outcome("solved", True, True, 1.0, 2, 1000),
        _outcome("solved", True, False, 1.0, 4, 2000),
        _outcome("budget_exhausted", False, False, 0.3, 10, 5000),
    ]
    m = metrics.aggregate_cell(outs)
    assert m.n_runs == 3
    assert m.exact_reconstruction_rate == pytest.approx(2 / 3)
    assert m.ood_exact_rate == pytest.approx(1 / 3)
    assert m.budget_exhaustion_rate == pytest.approx(1 / 3)
    assert m.mean_iterations_to_solve == pytest.approx(3.0)  # only solved runs


def test_score_run_fills_fields():
    t = _mod_transform()
    test_id = sampler.make_samples(t, 100, 1, "test_id")
    test_ood = sampler.make_samples(t, 100, 1, "test_ood")
    cell = GridCell(n=1, m=1, k=10, corpus="B", complexity_band="low", noise="none",
                    mode="scratch", model_id="mock", query_budget=0)
    o = RunOutcome(task_id="t", cell=cell, repeat_idx=0, final_source=t.source,
                   outcome="solved", iterations_used=1, tokens_used=10, usd_cost=0.0,
                   transcript_path="x.json")
    scored = metrics.score_run(o, t, test_id, test_ood)
    assert scored.exact_pass and scored.ood_exact_pass
    assert scored.row_accuracy == 1.0


def test_ast_similarity_exploratory():
    src = "def transform(x):\n    return {'out_0': x['in_0'] % 4}"
    assert metrics.ast_similarity(src, src) == 1.0
    assert 0.0 <= metrics.ast_similarity("def transform(x):\n    return {}", src) < 1.0
    assert metrics.ast_similarity("def (((", src) == 0.0  # unparseable -> 0
