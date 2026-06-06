"""Contract tests for types.py (CHECKPOINT 0)."""

import pytest
from pydantic import ValidationError

from progrecon.types import (
    ComplexityTuple,
    DataflowGraph,
    Example,
    FieldSpec,
    GridCell,
    OpNode,
    PerturbationSpec,
    RunOutcome,
    SampleSet,
    Schema,
    Transform,
)


def _mk_transform() -> Transform:
    s = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=10)],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="nd0", op="mod_k", params={"k": 3}, inputs=["in_0"])],
        output_map={"out_0": "nd0"},
    )
    ct = ComplexityTuple(n=1, m=1, depth=1, max_arity=1, max_tier=0, n_branches=0)
    return Transform(
        id="B-T-1",
        corpus="B",
        semantic=False,
        schema=s,
        source="def transform(x):\n    return {'out_0': x['in_0'] % 3}",
        graph=g,
        complexity=ct,
        px_seed=1,
    )


def test_transform_roundtrip_json():
    t = _mk_transform()
    j = t.model_dump_json()
    t2 = Transform.model_validate_json(j)
    assert t2 == t
    assert t2.schema.n == 1 and t2.schema.m == 1


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        FieldSpec(name="x", type="int", bogus=1)  # type: ignore[call-arg]


def test_schema_rejects_empty_and_duplicate():
    with pytest.raises(ValidationError):
        Schema(inputs=[], outputs=[FieldSpec(name="out_0", type="int")])
    with pytest.raises(ValidationError):
        Schema(
            inputs=[FieldSpec(name="a", type="int"), FieldSpec(name="a", type="int")],
            outputs=[FieldSpec(name="out_0", type="int")],
        )


def test_fieldspec_low_high_validation():
    with pytest.raises(ValidationError):
        FieldSpec(name="a", type="int", low=10, high=0)


def test_duplicate_node_ids_rejected():
    with pytest.raises(ValidationError):
        DataflowGraph(
            nodes=[
                OpNode(node_id="nd0", op="mod_k", params={"k": 3}, inputs=["in_0"]),
                OpNode(node_id="nd0", op="mod_k", params={"k": 5}, inputs=["in_0"]),
            ],
            output_map={"out_0": "nd0"},
        )


def test_sampleset_test_flags():
    train = SampleSet(kind="train", examples=[Example(row_id=0, x={"a": 1}, y={"b": 2})], seed=1)
    test = SampleSet(kind="test_id", examples=[Example(row_id=5, x={"a": 1}, y={"b": 2})], seed=2)
    assert not train.is_test and test.is_test
    assert test.row_ids == {5}


def test_gridcell_key_is_stable():
    c = GridCell(
        n=3, m=1, k=100, corpus="B", complexity_band="med", noise="none",
        mode="scratch", model_id="cheap", query_budget=0,
    )
    assert c.cell_key() == "n3-m1-k100-B-med-none-scratch-cheap-q0"


def test_perturbation_rejects_negative_distractors():
    with pytest.raises(ValidationError):
        PerturbationSpec(seed=1, n_distractors=-1)


def test_runoutcome_defaults_scoring_none():
    cell = GridCell(
        n=3, m=1, k=100, corpus="B", complexity_band="med", noise="none",
        mode="scratch", model_id="cheap", query_budget=0,
    )
    ro = RunOutcome(
        task_id="t", cell=cell, repeat_idx=0, final_source=None,
        outcome="error", iterations_used=0, tokens_used=0, usd_cost=0.0,
        transcript_path="x.json",
    )
    assert ro.exact_pass is None and ro.row_accuracy is None
