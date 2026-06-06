"""Semantic <-> abstract twin construction.

``make_abstract`` takes a (semantic) transform and produces a structurally
identical abstract twin: it masks field names/labels and swaps each op for a
*different* op of the SAME tier, arity, and branch-arm count that produces the
SAME output type — so the dataflow topology and the entire ComplexityTuple are
preserved while the specific computation (and any semantic prior) changes. The
construction asserts structural isomorphism and equal complexity tuples.

``make_pseudo_semantic`` is the Claim-3 control: it dresses an abstract
transform in *misleading* semantic labels that do not correspond to the
computation.
"""

from __future__ import annotations

import random

from ..dsl import codegen
from ..dsl.graph import compute_complexity_tuple, node_out_types, validate_or_raise
from ..dsl.ops import BRANCH_OPS, branch_arms
from ..dsl.ops_spec import OPS_SPEC, infer_out_type, op_accepts
from ..types import DataflowGraph, FieldSpec, OpNode, Schema, Transform
from .generator import reconstruct_producer, sample_params

# Misleading economic vocabulary for the pseudo-semantic control.
_PSEUDO_INPUT_LABELS = ["base_price", "tax_rate", "quantity", "discount", "tenure", "region_code", "tier", "bonus"]
_PSEUDO_OUTPUT_LABELS = ["net_total", "fee", "score", "rebate", "premium"]


def _canonicalize_names(schema: Schema, g: DataflowGraph) -> tuple[Schema, DataflowGraph]:
    in_map = {f.name: f"in_{i}" for i, f in enumerate(schema.inputs)}
    out_map = {f.name: f"out_{i}" for i, f in enumerate(schema.outputs)}
    new_inputs = [
        f.model_copy(update={"name": in_map[f.name], "semantic_label": None})
        for f in schema.inputs
    ]
    new_outputs = [
        f.model_copy(update={"name": out_map[f.name], "semantic_label": None})
        for f in schema.outputs
    ]
    new_nodes = [
        OpNode(node_id=nd.node_id, op=nd.op, params=nd.params, inputs=[in_map.get(i, i) for i in nd.inputs])
        for nd in g.nodes
    ]
    new_output_map = {out_map[k]: v for k, v in g.output_map.items()}
    return (
        Schema(inputs=new_inputs, outputs=new_outputs),
        DataflowGraph(nodes=new_nodes, output_map=new_output_map),
    )


def _candidate_ops(op: str, arity: int, in_types: list[str]) -> list[str]:
    spec = OPS_SPEC[op]
    is_branch = op in BRANCH_OPS
    out: list[str] = []
    for name, s in OPS_SPEC.items():
        if name == op or s.tier != spec.tier:
            continue
        if (name in BRANCH_OPS) != is_branch:
            continue  # preserve branchiness (and thus n_branches)
        if s.variadic:
            if not (s.min_arity <= arity <= s.max_arity):
                continue
        elif s.min_arity != arity:
            continue
        if not op_accepts(name, in_types):  # type: ignore[arg-type]
            continue
        out.append(name)
    return out


def _swap_ops(schema: Schema, g: DataflowGraph, rng: random.Random) -> DataflowGraph:
    out_types = node_out_types(g, schema)
    field_types = {f.name: f.type for f in schema.inputs}
    new_nodes: list[OpNode] = []
    for nd in g.nodes:
        in_types = [out_types.get(i) or field_types.get(i) for i in nd.inputs]
        want_out = infer_out_type(nd.op, in_types, nd.params)  # type: ignore[arg-type]
        want_arms = branch_arms(nd.op, nd.params)
        cands = _candidate_ops(nd.op, len(nd.inputs), in_types)  # type: ignore[arg-type]
        rng.shuffle(cands)
        chosen_op, chosen_params = nd.op, nd.params
        for cand in cands:
            producers = [reconstruct_producer(i, schema, out_types) for i in nd.inputs]
            try:
                params = sample_params(cand, producers, rng)
            except Exception:  # noqa: BLE001
                continue
            if infer_out_type(cand, in_types, params) != want_out:  # type: ignore[arg-type]
                continue
            if branch_arms(cand, params) != want_arms:
                continue
            chosen_op, chosen_params = cand, params
            break
        new_nodes.append(OpNode(node_id=nd.node_id, op=chosen_op, params=chosen_params, inputs=nd.inputs))
    return DataflowGraph(nodes=new_nodes, output_map=dict(g.output_map))


def _assert_isomorphic(a: DataflowGraph, b: DataflowGraph) -> None:
    a_ids = [nd.node_id for nd in a.nodes]
    b_ids = [nd.node_id for nd in b.nodes]
    assert a_ids == b_ids, "twin node ids differ"
    a_edges = {nd.node_id: list(nd.inputs) for nd in a.nodes}
    b_edges = {nd.node_id: list(nd.inputs) for nd in b.nodes}
    assert a_edges == b_edges, "twin edges differ (topology not preserved)"
    assert a.output_map == b.output_map, "twin output_map differs"


def make_abstract(transform: Transform, *, seed: int, twin_id: str | None = None) -> Transform:
    """Produce the structurally-identical abstract twin of a transform."""
    rng = random.Random(seed)
    schema, g = _canonicalize_names(transform.schema, transform.graph)
    twin_graph = _swap_ops(schema, g, rng)

    _assert_isomorphic(g, twin_graph)
    validate_or_raise(twin_graph, schema)
    ct = compute_complexity_tuple(twin_graph, schema)
    assert ct == transform.complexity, (
        f"twin complexity {ct} != base {transform.complexity}"
    )
    source = codegen.graph_to_source(twin_graph, schema)
    new_id = twin_id or f"{transform.id}~twin"
    prov = dict(transform.provenance)
    prov["twin_of"] = transform.id
    prov["twin_seed"] = seed
    return transform.model_copy(
        update={
            "id": new_id,
            "semantic": False,
            "domain": None,
            "description": None,
            "schema": schema,
            "graph": twin_graph,
            "complexity": ct,
            "source": source,
            "twin_id": transform.id,
            "provenance": prov,
        }
    )


def make_pseudo_semantic(transform: Transform, *, seed: int) -> Transform:
    """Claim-3 control: attach MISLEADING semantic labels to an abstract transform."""
    rng = random.Random(seed)
    in_labels = list(_PSEUDO_INPUT_LABELS)
    out_labels = list(_PSEUDO_OUTPUT_LABELS)
    rng.shuffle(in_labels)
    rng.shuffle(out_labels)
    new_inputs: list[FieldSpec] = []
    for i, f in enumerate(transform.schema.inputs):
        label = in_labels[i % len(in_labels)]
        new_inputs.append(f.model_copy(update={"semantic_label": label}))
    new_outputs: list[FieldSpec] = []
    for i, f in enumerate(transform.schema.outputs):
        label = out_labels[i % len(out_labels)]
        new_outputs.append(f.model_copy(update={"semantic_label": label}))
    schema = Schema(inputs=new_inputs, outputs=new_outputs)
    prov = dict(transform.provenance)
    prov["pseudo_semantic_of"] = transform.id
    return transform.model_copy(
        update={
            "id": f"{transform.id}~pseudo",
            "semantic": True,
            "domain": "pseudo",
            "description": "Control: semantic labels are deliberately unrelated to the computation.",
            "schema": schema,
            "provenance": prov,
        }
    )
