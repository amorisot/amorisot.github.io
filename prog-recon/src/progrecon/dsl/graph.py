"""Dataflow-graph validation, topological ordering, and complexity tuple.

The graph is the canonical representation of a transform. ``Transform.source``
is codegen'd from it (see ``dsl.codegen``); the executor evaluates it directly.
Validation enforces the structural rules from the build spec §4:
no cycles, all inputs resolve, fan-in matches op arity, all outputs mapped,
no dead nodes, no constant (input-independent) outputs, type-compatible edges.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..types import ComplexityTuple, DataflowGraph, Schema
from . import ops
from .ops_spec import OPS_SPEC, ConcreteType, infer_out_type, op_accepts


class GraphError(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    message: str
    node_id: str | None = None

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        loc = f" [{self.node_id}]" if self.node_id else ""
        return f"{self.code}{loc}: {self.message}"


class GraphValidationError(Exception):
    def __init__(self, errors: list[GraphError]):
        self.errors = errors
        super().__init__("; ".join(str(e) for e in errors))


def _node_deps(g: DataflowGraph) -> dict[str, list[str]]:
    """node_id -> list of upstream node_ids it depends on (field inputs ignored)."""
    node_ids = {nd.node_id for nd in g.nodes}
    return {nd.node_id: [i for i in nd.inputs if i in node_ids] for nd in g.nodes}


def topological_order(g: DataflowGraph) -> list[str]:
    """Kahn's algorithm. Raises GraphValidationError on a cycle."""
    deps = _node_deps(g)
    indeg = {nid: len(ds) for nid, ds in deps.items()}
    dependents: dict[str, list[str]] = {nid: [] for nid in deps}
    for nid, ds in deps.items():
        for d in ds:
            dependents[d].append(nid)
    queue = sorted([nid for nid, d in indeg.items() if d == 0])
    order: list[str] = []
    while queue:
        nid = queue.pop(0)
        order.append(nid)
        for dep in dependents[nid]:
            indeg[dep] -= 1
            if indeg[dep] == 0:
                queue.append(dep)
        queue.sort()
    if len(order) != len(g.nodes):
        remaining = sorted(set(deps) - set(order))
        raise GraphValidationError(
            [GraphError(code="cycle", message=f"cycle among nodes {remaining}")]
        )
    return order


def _input_field_types(schema: Schema) -> dict[str, ConcreteType]:
    return {f.name: f.type for f in schema.inputs}  # type: ignore[misc]


def node_out_types(g: DataflowGraph, schema: Schema) -> dict[str, ConcreteType]:
    """Concrete output type of every node, computed in topological order."""
    field_types = _input_field_types(schema)
    order = topological_order(g)
    out: dict[str, ConcreteType] = {}
    for nid in order:
        nd = g.node(nid)
        in_types: list[ConcreteType] = []
        for inp in nd.inputs:
            if inp in out:
                in_types.append(out[inp])
            elif inp in field_types:
                in_types.append(field_types[inp])
            else:
                # unresolved input; validate() reports it. Use a placeholder.
                in_types.append("int")
        out[nid] = infer_out_type(nd.op, in_types, nd.params)
    return out


def ancestor_nodes(g: DataflowGraph, roots: set[str]) -> set[str]:
    """All node_ids in the transitive input-cone of `roots` (inclusive)."""
    by_id = {nd.node_id: nd for nd in g.nodes}
    keep: set[str] = set()
    stack = [r for r in roots if r in by_id]
    while stack:
        nid = stack.pop()
        if nid in keep:
            continue
        keep.add(nid)
        stack.extend(i for i in by_id[nid].inputs if i in by_id)
    return keep


def _reaches_input(g: DataflowGraph, start: str, field_names: set[str]) -> bool:
    """Does node `start` transitively depend on at least one input field?"""
    seen: set[str] = set()
    stack = [start]
    node_ids = {nd.node_id for nd in g.nodes}
    while stack:
        nid = stack.pop()
        nd = g.node(nid)
        for inp in nd.inputs:
            if inp in field_names:
                return True
            if inp in node_ids and inp not in seen:
                seen.add(inp)
                stack.append(inp)
    return False


def validate(g: DataflowGraph, schema: Schema) -> list[GraphError]:
    """Return a list of structural/type errors (empty list == valid)."""
    errors: list[GraphError] = []
    node_ids = {nd.node_id for nd in g.nodes}
    field_names = {f.name for f in schema.inputs}

    # 1. Unknown ops + input resolution + arity.
    for nd in g.nodes:
        if nd.op not in OPS_SPEC:
            errors.append(GraphError(code="unknown_op", message=f"unknown op {nd.op!r}", node_id=nd.node_id))
            continue
        spec = OPS_SPEC[nd.op]
        for inp in nd.inputs:
            if inp not in node_ids and inp not in field_names:
                errors.append(
                    GraphError(code="bad_input", message=f"input {inp!r} is neither a node nor a field", node_id=nd.node_id)
                )
        arity = len(nd.inputs)
        if spec.variadic:
            if not (spec.min_arity <= arity <= spec.max_arity):
                errors.append(GraphError(code="arity", message=f"{nd.op} arity {arity} outside [{spec.min_arity},{spec.max_arity}]", node_id=nd.node_id))
        elif arity != spec.min_arity:
            errors.append(GraphError(code="arity", message=f"{nd.op} expects {spec.min_arity} inputs, got {arity}", node_id=nd.node_id))

    # 2. Cycle check (and we need an order for type checks).
    try:
        order = topological_order(g)
    except GraphValidationError as e:
        return errors + e.errors

    # Type inference is only meaningful once structure is sound.
    can_type = not any(e.code in {"unknown_op", "bad_input", "arity"} for e in errors)

    # 3. Type compatibility along edges (needs resolvable inputs).
    if can_type:
        out_types = node_out_types(g, schema)
        field_types = _input_field_types(schema)
        for nid in order:
            nd = g.node(nid)
            in_types = [out_types.get(i) or field_types.get(i) for i in nd.inputs]
            if any(t is None for t in in_types):
                continue
            if not op_accepts(nd.op, [t for t in in_types if t is not None]):
                errors.append(GraphError(code="type", message=f"{nd.op} cannot accept input types {in_types}", node_id=nid))

    # 4. Outputs all mapped, to existing nodes, with compatible types.
    out_types_full = node_out_types(g, schema) if (node_ids and can_type) else {}
    for f in schema.outputs:
        if f.name not in g.output_map:
            errors.append(GraphError(code="unmapped_output", message=f"output {f.name!r} not in output_map"))
            continue
        producer = g.output_map[f.name]
        if producer not in node_ids:
            errors.append(GraphError(code="bad_output", message=f"output {f.name!r} maps to unknown node {producer!r}"))
            continue
        prod_t = out_types_full.get(producer)
        if prod_t is not None and not _type_satisfies_field(prod_t, f.type):  # type: ignore[arg-type]
            errors.append(GraphError(code="output_type", message=f"output {f.name!r} declared {f.type} but producer yields {prod_t}", node_id=producer))

    # 5. No dead nodes (every node feeds some output).
    live = set(g.output_map.values())
    deps = _node_deps(g)
    frontier = list(live)
    while frontier:
        nid = frontier.pop()
        for d in deps.get(nid, []):
            if d not in live:
                live.add(d)
                frontier.append(d)
    for nd in g.nodes:
        if nd.node_id not in live:
            errors.append(GraphError(code="dead_node", message="node feeds no output", node_id=nd.node_id))

    # 6. No constant (input-independent) outputs.
    for f in schema.outputs:
        producer = g.output_map.get(f.name)
        if producer in node_ids and not _reaches_input(g, producer, field_names):
            errors.append(GraphError(code="constant_output", message=f"output {f.name!r} does not depend on any input", node_id=producer))

    return errors


def _type_satisfies_field(produced: ConcreteType, declared: ConcreteType) -> bool:
    """int/float are interchangeable for a numeric field declaration."""
    numeric = {"int", "float"}
    if produced in numeric and declared in numeric:
        return True
    return produced == declared


def validate_or_raise(g: DataflowGraph, schema: Schema) -> None:
    errors = validate(g, schema)
    if errors:
        raise GraphValidationError(errors)


def compute_complexity_tuple(g: DataflowGraph, schema: Schema) -> ComplexityTuple:
    """Structural complexity tuple from the graph (n includes distractor fields)."""
    order = topological_order(g)
    depth_of: dict[str, int] = {}
    for nid in order:
        nd = g.node(nid)
        in_depths = [depth_of[i] for i in nd.inputs if i in depth_of]
        # field inputs contribute depth 0
        depth_of[nid] = 1 + (max(in_depths) if in_depths else 0)
    depth = max(depth_of.values()) if depth_of else 0
    max_arity = max((len(nd.inputs) for nd in g.nodes), default=0)
    max_tier = max((OPS_SPEC[nd.op].tier for nd in g.nodes), default=0)
    n_branches = sum(ops.branch_arms(nd.op, nd.params) for nd in g.nodes)
    return ComplexityTuple(
        n=schema.n, m=schema.m, depth=depth, max_arity=max_arity,
        max_tier=max_tier, n_branches=n_branches,
    )
