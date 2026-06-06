"""Corpus B generator — sample abstract Transforms from the DSL.

``sample_transform`` builds a dataflow DAG that hits target shape parameters
``(n, m, depth)`` using a generator *personality* (allowed tiers + topology
prior, drawn from ``config.yaml`` — NOT 50 copies of code). It wires only
type-compatible edges, samples op constants, prunes dead nodes, validates
against ``graph.validate``, codegen's the source, and computes the complexity
tuple. Degenerate graphs are rejected and retried with a fresh seed offset.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field

from ..config import GeneratorPersonality
from ..dsl import codegen
from ..dsl.graph import compute_complexity_tuple, validate
from ..dsl.ops_spec import OPS_SPEC, ConcreteType, infer_out_type, op_accepts, ops_by_tier
from ..types import DataflowGraph, FieldSpec, OpNode, Schema, Transform

_CATEGORIES = ["red", "green", "blue", "amber", "violet", "cyan"]


@dataclass
class Producer:
    """A value source available to the builder (an input field or a node)."""

    ref: str  # field name or node_id
    type: ConcreteType
    lo: float = -50.0  # coarse numeric range estimate (for param sampling)
    hi: float = 50.0
    categories: list[str] | None = None


@dataclass
class _BuildState:
    producers: list[Producer] = dc_field(default_factory=list)
    nodes: list[OpNode] = dc_field(default_factory=list)
    counter: int = 0


class GenerationError(Exception):
    pass


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------


def _make_inputs(n: int, rng: random.Random, with_categorical: bool) -> list[Producer]:
    prods: list[Producer] = []
    n_cat = 1 if (with_categorical and n >= 1) else 0
    for i in range(n):
        name = f"in_{i}"
        if i < n_cat:
            cats = rng.sample(_CATEGORIES, k=rng.randint(2, 4))
            prods.append(Producer(ref=name, type="categorical", categories=cats))
        else:
            lo = float(rng.randint(-20, 5))
            hi = lo + float(rng.randint(10, 80))
            prods.append(Producer(ref=name, type="int", lo=lo, hi=hi))
    return prods


def _fieldspec_for_input(p: Producer) -> FieldSpec:
    if p.type == "categorical":
        return FieldSpec(name=p.ref, type="categorical", categories=list(p.categories or []))
    return FieldSpec(name=p.ref, type=p.type, low=p.lo, high=p.hi)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Param sampling + range estimation
# ---------------------------------------------------------------------------


def sample_params(op: str, chosen: list[Producer], rng: random.Random) -> dict:
    if op == "mod_k":
        return {"k": rng.randint(2, 12)}
    if op == "round_to_k":
        return {"k": rng.choice([2, 5, 10, 25])}
    if op in ("digit_reverse", "abs_val", "negate", "sum_inputs", "product", "diff",
              "argmax_index", "median", "max_val", "min_val"):
        return {}
    if op == "clip":
        lo = rng.randint(-10, 5)
        return {"lo": lo, "hi": lo + rng.randint(5, 40)}
    if op == "affine":
        return {"a": rng.choice([-3, -2, -1, 1, 2, 3]), "b": rng.randint(-10, 10)}
    if op == "weighted_sum":
        return {"w": [rng.choice([-2, -1, 1, 2, 3]) for _ in chosen]}
    if op == "select_if":
        cond = chosen[0]
        if cond.type == "categorical" and cond.categories:
            k = max(1, len(cond.categories) // 2)
            return {"truthy": rng.sample(cond.categories, k=k)}
        return {}
    if op == "count_above":
        flo, fhi = _merged_range(chosen)
        return {"thr": rng.randint(int(flo), max(int(flo) + 1, int(fhi)))}
    if op == "bracket_dispatch":
        flo, fhi = _merged_range(chosen)
        lo_i, hi_i = int(flo), max(int(flo) + 3, int(fhi))
        n_thr = rng.randint(1, 3)
        cuts = sorted(rng.sample(range(lo_i + 1, hi_i), k=min(n_thr, max(1, hi_i - lo_i - 1))))
        return {"thresholds": cuts, "values": [rng.randint(0, 9) for _ in range(len(cuts) + 1)]}
    if op == "lookup_table":
        cats = chosen[0].categories or []
        return {"table": {c: rng.randint(0, 9) for c in cats}}
    raise GenerationError(f"no param sampler for op {op!r}")


def reconstruct_producer(ref: str, schema: Schema, out_types: Mapping[str, str]) -> Producer:
    """Rebuild a Producer for an input field or node ref (for re-sampling params).

    Shared by the perturbation and twin layers so they re-derive producer types
    and ranges identically.
    """
    for f in schema.inputs:
        if f.name == ref:
            return Producer(
                ref=ref,
                type=f.type,  # type: ignore[arg-type]
                lo=float(f.low) if f.low is not None else -50.0,
                hi=float(f.high) if f.high is not None else 50.0,
                categories=list(f.categories) if f.categories else None,
            )
    return Producer(ref=ref, type=out_types.get(ref, "int"), lo=-50.0, hi=50.0, categories=None)  # type: ignore[arg-type]


def _merged_range(prods: list[Producer]) -> tuple[float, float]:
    los = [p.lo for p in prods if p.type in ("int", "float")]
    his = [p.hi for p in prods if p.type in ("int", "float")]
    if not los:
        return (-50.0, 50.0)
    return (min(los), max(his))


def _estimate_range(op: str, params: dict, chosen: list[Producer]) -> tuple[float, float]:
    lo, hi = _merged_range(chosen)
    if op == "mod_k":
        return (0.0, float(params["k"] - 1))
    if op == "clip":
        return (float(params["lo"]), float(params["hi"]))
    if op == "abs_val":
        return (0.0, max(abs(lo), abs(hi)))
    if op == "negate":
        return (-hi, -lo)
    if op in ("argmax_index", "count_above"):
        return (0.0, float(len(chosen)))
    if op == "bracket_dispatch":
        vals = params["values"]
        return (float(min(vals)), float(max(vals)))
    if op == "lookup_table":
        vals = list(params["table"].values()) or [0]
        return (float(min(vals)), float(max(vals)))
    if op == "affine":
        a = params["a"]
        ends = [a * lo + params["b"], a * hi + params["b"]]
        return (min(ends), max(ends))
    return (lo, hi)


def _categories_of(op: str, params: dict, chosen: list[Producer]) -> list[str] | None:
    if op == "select_if":
        return chosen[1].categories if chosen[1].type == "categorical" else None
    if op == "bracket_dispatch":
        vals = params["values"]
        return [v for v in vals if isinstance(v, str)] or None
    if op == "lookup_table":
        vals = list(params["table"].values())
        return [v for v in vals if isinstance(v, str)] or None
    return None


# ---------------------------------------------------------------------------
# Node wiring
# ---------------------------------------------------------------------------


def _choose_inputs(
    op: str, prods: list[Producer], rng: random.Random, prefer_recent: bool
) -> list[Producer] | None:
    """Pick a type-compatible set of producers for `op`, or None if impossible."""
    spec = OPS_SPEC[op]

    if op == "select_if":
        # (cond, a, b): a,b must share a type; cond is any producer.
        by_type: dict[ConcreteType, list[Producer]] = {}
        for p in prods:
            by_type.setdefault(p.type, []).append(p)
        usable = [t for t, ps in by_type.items() if len(ps) >= 2]
        if not usable:
            return None
        t = rng.choice(usable)
        a, b = rng.sample(by_type[t], k=2)
        cond = rng.choice(prods)
        return [cond, a, b]

    if op == "lookup_table":
        cats = [p for p in prods if p.type == "categorical" and p.categories]
        return [rng.choice(cats)] if cats else None

    # Generic numeric/int ops.
    if spec.in_class == "int":
        pool = [p for p in prods if p.type == "int"]
    elif spec.in_class == "num":
        pool = [p for p in prods if p.type in ("int", "float")]
    elif spec.in_class == "categorical":
        pool = [p for p in prods if p.type == "categorical"]
    else:
        pool = list(prods)
    if not pool:
        return None

    if spec.variadic:
        max_a = min(spec.max_arity, len(pool))
        if max_a < spec.min_arity:
            return None
        arity = rng.randint(spec.min_arity, max_a)
    else:
        arity = spec.min_arity
        if len(pool) < arity:
            return None
    chosen = rng.sample(pool, k=arity)
    if prefer_recent and pool:
        recent = pool[-1]
        if recent not in chosen:
            chosen[0] = recent
    return chosen


def _add_node(
    state: _BuildState, allowed_ops: list[str], rng: random.Random, prefer_recent: bool
) -> bool:
    candidates = allowed_ops[:]
    rng.shuffle(candidates)
    for op in candidates:
        chosen = _choose_inputs(op, state.producers, rng, prefer_recent)
        if chosen is None:
            continue
        in_types = [p.type for p in chosen]
        if not op_accepts(op, in_types):
            continue
        params = sample_params(op, chosen, rng)
        out_type = infer_out_type(op, in_types, params)
        node_id = f"n{state.counter}"
        state.counter += 1
        state.nodes.append(OpNode(node_id=node_id, op=op, params=params, inputs=[p.ref for p in chosen]))
        lo, hi = _estimate_range(op, params, chosen)
        state.producers.append(
            Producer(ref=node_id, type=out_type, lo=lo, hi=hi, categories=_categories_of(op, params, chosen))
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Pruning + output selection
# ---------------------------------------------------------------------------


def _ancestors(nodes: list[OpNode], roots: set[str]) -> set[str]:
    by_id = {nd.node_id: nd for nd in nodes}
    keep: set[str] = set()
    stack = list(roots)
    while stack:
        nid = stack.pop()
        if nid not in by_id or nid in keep:
            continue
        keep.add(nid)
        stack.extend(i for i in by_id[nid].inputs if i in by_id)
    return keep


def _build_once(
    n: int, m: int, target_depth: int, personality: GeneratorPersonality, seed: int
) -> tuple[Schema, DataflowGraph] | None:
    rng = random.Random(seed)
    allowed_ops = ops_by_tier(personality.tiers)
    with_cat = any(OPS_SPEC[o].in_class == "categorical" for o in allowed_ops) and rng.random() < 0.6
    state = _BuildState(producers=_make_inputs(n, rng, with_cat))

    prefer_recent = personality.arity_bias != "wide"
    target_nodes = max(m, target_depth, 2)
    # Build a primary chain (prefer_recent) to reach depth, then breadth.
    budget = 0
    max_budget = 60
    while len([nd for nd in state.nodes]) < target_nodes + m and budget < max_budget:
        budget += 1
        pr = prefer_recent if budget <= target_depth else (rng.random() < 0.4)
        if not _add_node(state, allowed_ops, rng, pr):
            # try once more without recency preference
            if not _add_node(state, allowed_ops, rng, False):
                break

    node_producers = [p for p in state.producers if p.ref.startswith("n")]
    if len(node_producers) < m:
        return None

    # Pick m distinct output producers, preferring the latest (deepest) ones.
    outs = node_producers[-m:] if m <= len(node_producers) else node_producers
    output_map = {f"out_{i}": p.ref for i, p in enumerate(outs)}

    keep = _ancestors(state.nodes, set(output_map.values()))
    kept_nodes = [nd for nd in state.nodes if nd.node_id in keep]

    inputs = [p for p in state.producers if not p.ref.startswith("n")]
    # Only keep input fields actually referenced (others would be dead inputs);
    # distractors are added later by the perturbation layer, not here.
    referenced = {i for nd in kept_nodes for i in nd.inputs}
    used_inputs = [p for p in inputs if p.ref in referenced]
    if not used_inputs:
        return None
    # Re-index input field names to be contiguous in_0..in_{k-1}.
    rename = {p.ref: f"in_{j}" for j, p in enumerate(used_inputs)}
    kept_nodes = [
        OpNode(
            node_id=nd.node_id,
            op=nd.op,
            params=nd.params,
            inputs=[rename.get(i, i) for i in nd.inputs],
        )
        for nd in kept_nodes
    ]
    schema_inputs = [_fieldspec_for_input(Producer(ref=rename[p.ref], type=p.type, lo=p.lo, hi=p.hi, categories=p.categories)) for p in used_inputs]

    # Output field types from producer types.
    prod_by_ref = {p.ref: p for p in node_producers}
    schema_outputs = []
    for i, p in enumerate(outs):
        t = prod_by_ref[p.ref].type
        decl: ConcreteType = "float" if t == "float" else t
        schema_outputs.append(FieldSpec(name=f"out_{i}", type=decl))

    schema = Schema(inputs=schema_inputs, outputs=schema_outputs)
    g = DataflowGraph(nodes=kept_nodes, output_map=output_map)
    return schema, g


def sample_transform(
    personality: GeneratorPersonality,
    n: int,
    m: int,
    target_depth: int,
    seed: int,
    *,
    transform_id: str,
    max_attempts: int = 40,
) -> Transform:
    """Generate one valid Corpus-B Transform hitting (n, m, ~depth)."""
    for attempt in range(max_attempts):
        built = _build_once(n, m, target_depth, personality, seed + attempt * 7919)
        if built is None:
            continue
        schema, g = built
        if validate(g, schema):
            continue
        ct = compute_complexity_tuple(g, schema)
        source = codegen.graph_to_source(g, schema)
        return Transform(
            id=transform_id,
            corpus="B",
            semantic=False,
            domain=None,
            description=None,
            schema=schema,
            source=source,
            graph=g,
            complexity=ct,
            px_seed=seed,
            provenance={
                "generator": personality.name,
                "tiers": personality.tiers,
                "seed": seed,
                "attempt": attempt,
                "target": {"n": n, "m": m, "depth": target_depth},
            },
        )
    raise GenerationError(
        f"failed to generate a valid transform for {transform_id} after {max_attempts} attempts "
        f"(personality={personality.name}, n={n}, m={m}, depth={target_depth})"
    )
