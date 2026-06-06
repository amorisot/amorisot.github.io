"""Executor + graph validation + codegen round-trip tests (CHECKPOINT 2)."""

import random

import pytest

from progrecon.dsl import codegen, executor, graph
from progrecon.types import DataflowGraph, FieldSpec, OpNode, Schema


def _schema(n_int=3, out_type="float"):
    return Schema(
        inputs=[FieldSpec(name=f"in_{i}", type="int", low=0, high=50) for i in range(n_int)],
        outputs=[FieldSpec(name="out_0", type=out_type)],
    )


def _good_graph():
    return DataflowGraph(
        nodes=[
            OpNode(node_id="n0", op="mod_k", params={"k": 3}, inputs=["in_0"]),
            OpNode(node_id="n1", op="weighted_sum", params={"w": [1.0, 2.0]}, inputs=["in_1", "in_2"]),
            OpNode(node_id="n2", op="sum_inputs", params={}, inputs=["n0", "n1"]),
        ],
        output_map={"out_0": "n2"},
    )


def test_valid_graph_has_no_errors():
    assert graph.validate(_good_graph(), _schema()) == []


def test_codegen_roundtrip_matches_run_graph():
    s, g = _schema(), _good_graph()
    src = codegen.graph_to_source(g, s)
    rng = random.Random(7)
    for _ in range(300):
        x = {f.name: rng.randint(0, 50) for f in s.inputs}
        assert executor.run_graph(g, x) == executor.run_source(src, x)


def test_complexity_tuple():
    ct = graph.compute_complexity_tuple(_good_graph(), _schema())
    assert (ct.n, ct.m, ct.depth, ct.max_arity, ct.max_tier, ct.n_branches) == (3, 1, 2, 2, 1, 0)


def test_branch_complexity_counts_arms():
    s = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=100)],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[
            OpNode(node_id="n0", op="bracket_dispatch", params={"thresholds": [10, 20], "values": [0, 1, 2]}, inputs=["in_0"]),
        ],
        output_map={"out_0": "n0"},
    )
    assert graph.validate(g, s) == []
    ct = graph.compute_complexity_tuple(g, s)
    assert ct.max_tier == 3 and ct.n_branches == 3


def test_detects_cycle():
    s = _schema(1)
    g = DataflowGraph(
        nodes=[
            OpNode(node_id="n0", op="diff", params={}, inputs=["in_0", "n1"]),
            OpNode(node_id="n1", op="diff", params={}, inputs=["in_0", "n0"]),
        ],
        output_map={"out_0": "n0"},
    )
    codes = {e.code for e in graph.validate(g, s)}
    assert "cycle" in codes


def test_detects_missing_input():
    s = _schema(1)
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="mod_k", params={"k": 3}, inputs=["nope"])],
        output_map={"out_0": "n0"},
    )
    codes = {e.code for e in graph.validate(g, s)}
    assert "bad_input" in codes


def test_detects_unmapped_output():
    s = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=9)],
        outputs=[FieldSpec(name="out_0", type="int"), FieldSpec(name="out_1", type="int")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="mod_k", params={"k": 3}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    codes = {e.code for e in graph.validate(g, s)}
    assert "unmapped_output" in codes


def test_detects_dead_node():
    s = _schema(2, out_type="int")
    g = DataflowGraph(
        nodes=[
            OpNode(node_id="n0", op="mod_k", params={"k": 3}, inputs=["in_0"]),
            OpNode(node_id="n_dead", op="mod_k", params={"k": 5}, inputs=["in_1"]),
        ],
        output_map={"out_0": "n0"},
    )
    codes = {e.code for e in graph.validate(g, s)}
    assert "dead_node" in codes


def test_detects_arity_mismatch():
    s = _schema(2, out_type="int")
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="mod_k", params={"k": 3}, inputs=["in_0", "in_1"])],
        output_map={"out_0": "n0"},
    )
    codes = {e.code for e in graph.validate(g, s)}
    assert "arity" in codes


def test_detects_type_mismatch():
    # lookup_table needs categorical input but gets an int field.
    s = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=9)],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="lookup_table", params={"table": {"a": 1}}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    codes = {e.code for e in graph.validate(g, s)}
    assert "type" in codes


def test_detects_unknown_op():
    s = _schema(1)
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="not_an_op", params={}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    codes = {e.code for e in graph.validate(g, s)}
    assert "unknown_op" in codes


def test_validate_or_raise():
    s = _schema(1)
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="mod_k", params={"k": 3}, inputs=["nope"])],
        output_map={"out_0": "n0"},
    )
    with pytest.raises(graph.GraphValidationError):
        graph.validate_or_raise(g, s)
    # a valid one does not raise
    graph.validate_or_raise(_good_graph(), _schema())


def test_run_graph_missing_field_raises():
    g = _good_graph()
    with pytest.raises(executor.ExecutionError):
        executor.run_graph(g, {"in_0": 1})  # missing in_1, in_2
