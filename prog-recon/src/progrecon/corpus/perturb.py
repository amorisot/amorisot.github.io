"""Perturbation grammar (instancing) — protocol §3.6.

Six operators turn a base Transform into a concrete instance:
1. resample_constants  — re-draw op constants (type-preserving)
2. permute_fields      — reorder the input fields (presentation order)
3. rename_fields       — rename fields, updating all graph references
4. distribution_shift  — shift the task's P_x (numeric ranges)
5. n_distractors       — add irrelevant input fields no node consumes
                         (the clean n-lever for Claim 1)
6. branch_coverage_bias — a flag consumed by the sampler (no structural change)

All choices are driven by ``spec.seed``. The result is re-codegen'd from the
(possibly modified) graph and re-validated; source and graph never diverge.
"""

from __future__ import annotations

import copy
import random

from ..dsl import codegen
from ..dsl.graph import compute_complexity_tuple, node_out_types, validate_or_raise
from ..types import (
    DataflowGraph,
    FieldSpec,
    OpNode,
    PerturbationSpec,
    Schema,
    Task,
    Transform,
)
from .generator import reconstruct_producer, sample_params

_DISTRACTOR_CATS = ["alpha", "beta", "gamma", "delta"]


def _resample_constants(schema: Schema, g: DataflowGraph, rng: random.Random) -> DataflowGraph:
    out_types = node_out_types(g, schema)
    new_nodes: list[OpNode] = []
    for nd in g.nodes:
        chosen = [reconstruct_producer(i, schema, out_types) for i in nd.inputs]
        try:
            params = sample_params(nd.op, chosen, rng)
        except Exception:  # noqa: BLE001 - fall back to original constants
            params = nd.params
        new_nodes.append(OpNode(node_id=nd.node_id, op=nd.op, params=params, inputs=nd.inputs))
    return DataflowGraph(nodes=new_nodes, output_map=dict(g.output_map))


def _permute_fields(schema: Schema, rng: random.Random) -> Schema:
    inputs = list(schema.inputs)
    rng.shuffle(inputs)
    return Schema(inputs=inputs, outputs=list(schema.outputs))


def _rename_fields(schema: Schema, g: DataflowGraph, rng: random.Random) -> tuple[Schema, DataflowGraph]:
    in_names = [f.name for f in schema.inputs]
    out_names = [f.name for f in schema.outputs]
    new_in = [f"in_{i}" for i in range(len(in_names))]
    rng.shuffle(new_in)
    new_out = [f"out_{i}" for i in range(len(out_names))]
    rng.shuffle(new_out)
    in_map = dict(zip(in_names, new_in, strict=True))
    out_map = dict(zip(out_names, new_out, strict=True))

    new_inputs = [
        f.model_copy(update={"name": in_map[f.name]}) for f in schema.inputs
    ]
    new_outputs = [
        f.model_copy(update={"name": out_map[f.name]}) for f in schema.outputs
    ]
    new_nodes = [
        OpNode(
            node_id=nd.node_id,
            op=nd.op,
            params=nd.params,
            inputs=[in_map.get(i, i) for i in nd.inputs],
        )
        for nd in g.nodes
    ]
    new_output_map = {out_map[k]: v for k, v in g.output_map.items()}
    return (
        Schema(inputs=new_inputs, outputs=new_outputs),
        DataflowGraph(nodes=new_nodes, output_map=new_output_map),
    )


def _distribution_shift(schema: Schema, rng: random.Random, shift_factor: float) -> Schema:
    new_inputs: list[FieldSpec] = []
    for f in schema.inputs:
        if f.type in ("int", "float") and f.low is not None and f.high is not None:
            width = f.high - f.low
            shift = rng.uniform(-0.5, 0.5) * width
            new_low = f.low + shift
            new_high = new_low + width * shift_factor
            new_inputs.append(f.model_copy(update={"low": new_low, "high": new_high}))
        else:
            new_inputs.append(f)
    return Schema(inputs=new_inputs, outputs=list(schema.outputs))


def _add_distractors(schema: Schema, n: int, rng: random.Random) -> Schema:
    existing = {f.name for f in schema.inputs}
    new_inputs = list(schema.inputs)
    i = 0
    added = 0
    while added < n:
        name = f"distractor_{i}"
        i += 1
        if name in existing:
            continue
        if rng.random() < 0.3:
            cats = rng.sample(_DISTRACTOR_CATS, k=rng.randint(2, 4))
            new_inputs.append(FieldSpec(name=name, type="categorical", categories=cats))
        else:
            lo = float(rng.randint(-20, 5))
            new_inputs.append(FieldSpec(name=name, type="int", low=lo, high=lo + float(rng.randint(10, 80))))
        added += 1
    return Schema(inputs=new_inputs, outputs=list(schema.outputs))


def apply(transform: Transform, spec: PerturbationSpec) -> Transform:
    """Apply the perturbation spec to a base transform, returning a new one."""
    rng = random.Random(spec.seed)
    schema = copy.deepcopy(transform.schema)
    g = copy.deepcopy(transform.graph)

    if spec.resample_constants:
        g = _resample_constants(schema, g, rng)
    if spec.permute_fields:
        schema = _permute_fields(schema, rng)
    if spec.rename_fields:
        schema, g = _rename_fields(schema, g, rng)
    if spec.distribution_shift:
        # Use a fixed widen factor here; OOD test shift is separate (sampler).
        schema = _distribution_shift(schema, rng, shift_factor=1.0)
    if spec.n_distractors > 0:
        schema = _add_distractors(schema, spec.n_distractors, rng)

    validate_or_raise(g, schema)
    ct = compute_complexity_tuple(g, schema)
    source = codegen.graph_to_source(g, schema)
    prov = dict(transform.provenance)
    prov["perturbation"] = spec.model_dump()
    return transform.model_copy(
        update={
            "schema": schema,
            "graph": g,
            "complexity": ct,
            "source": source,
            "provenance": prov,
        }
    )


def make_task(transform: Transform, spec: PerturbationSpec) -> Task:
    """Apply a perturbation and wrap it as a Task with a derived task_id."""
    perturbed = apply(transform, spec)
    return Task(task_id=f"{transform.id}#p{spec.seed}", transform=perturbed, perturbation=spec)
