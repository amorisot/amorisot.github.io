"""Identifiability check tests (CHECKPOINT 3)."""

from progrecon.config import load_config
from progrecon.corpus import generator, identifiability
from progrecon.types import DataflowGraph, FieldSpec, OpNode, Schema

from .conftest import build_transform

CFG = load_config()


def test_well_identified_transform_passes():
    # clip(digit_reverse(in_0)) over a wide int range: not reducible to a single op.
    t = generator.sample_transform(CFG.generators[0], n=2, m=1, target_depth=2, seed=5, transform_id="B-ok")
    rep = identifiability.check(t, n_probes=400)
    assert rep.deterministic
    assert rep.passed, rep.flags


def test_constant_output_is_quarantined():
    schema = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=20)],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="clip", params={"lo": 5, "hi": 5}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    t = build_transform(schema, g, tid="B-const")
    rep = identifiability.check(t, n_probes=200)
    assert not rep.passed
    assert "constant_output" in rep.flags


def test_simpler_equivalent_is_quarantined():
    # affine(affine(in_0)) reduces to a single affine -> under-identified.
    schema = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=50)],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[
            OpNode(node_id="n0", op="affine", params={"a": 2, "b": 1}, inputs=["in_0"]),
            OpNode(node_id="n1", op="affine", params={"a": 3, "b": -4}, inputs=["n0"]),
        ],
        output_map={"out_0": "n1"},
    )
    t = build_transform(schema, g, tid="B-reducible")
    rep = identifiability.check(t, n_probes=200)
    assert not rep.passed
    assert "simpler_equivalent_exists" in rep.flags
    assert "out_0" in rep.simpler_equivalent_outputs


def test_incomplete_branch_coverage_is_quarantined():
    # bracket threshold above the input range -> the upper arm never fires.
    schema = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=10)],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="bracket_dispatch", params={"thresholds": [100], "values": [0, 1]}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    t = build_transform(schema, g, tid="B-uncovered")
    rep = identifiability.check(t, n_probes=200)
    assert not rep.passed
    assert "incomplete_branch_coverage" in rep.flags
    cov = next(bc for bc in rep.branch_coverage if bc.node_id == "n0")
    assert 1 in cov.missing


def test_full_branch_coverage_passes():
    schema = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=100)],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="bracket_dispatch", params={"thresholds": [33, 66], "values": [0, 1, 2]}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    t = build_transform(schema, g, tid="B-covered")
    rep = identifiability.check(t, n_probes=400)
    cov = next(bc for bc in rep.branch_coverage if bc.node_id == "n0")
    assert cov.missing == []
    assert rep.passed
