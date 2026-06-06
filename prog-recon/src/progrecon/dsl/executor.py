"""Canonical ground-truth evaluation.

``run_graph`` is THE canonical evaluator (topological eval over ``dsl.ops``).
``run_source`` executes a codegen'd ``Transform.source`` string IN-PROCESS and
is *trusted* — it is only ever used for ground-truth and for round-trip tests
that confirm codegen agrees with ``run_graph``. Untrusted (model-authored)
programs must go through ``harness.sandbox`` instead.
"""

from __future__ import annotations

from typing import Any

from ..types import DataflowGraph, Transform
from . import ops
from .graph import topological_order


class ExecutionError(Exception):
    pass


def run_graph(g: DataflowGraph, x: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the graph on input row x; returns the output dict y."""
    order = topological_order(g)
    cache: dict[str, Any] = {}
    for nid in order:
        nd = g.node(nid)
        argv: list[Any] = []
        for inp in nd.inputs:
            if inp in cache:
                argv.append(cache[inp])
            elif inp in x:
                argv.append(x[inp])
            else:
                raise ExecutionError(
                    f"node {nid!r}: input {inp!r} is neither a computed node nor present in x"
                )
        fn = ops.get_op(nd.op)
        cache[nid] = fn(nd.params, *argv)
    y: dict[str, Any] = {}
    for out_name, producer in g.output_map.items():
        if producer not in cache:
            raise ExecutionError(f"output {out_name!r} maps to uncomputed node {producer!r}")
        y[out_name] = cache[producer]
    return y


def run_graph_trace(g: DataflowGraph, x: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Like run_graph but also returns the full node-value cache (for tracing)."""
    order = topological_order(g)
    cache: dict[str, Any] = {}
    for nid in order:
        nd = g.node(nid)
        argv = [cache[i] if i in cache else x[i] for i in nd.inputs]
        cache[nid] = ops.get_op(nd.op)(nd.params, *argv)
    y = {out_name: cache[producer] for out_name, producer in g.output_map.items()}
    return y, cache


def run_source(source: str, x: dict[str, Any]) -> dict[str, Any]:
    """Execute a trusted source string in-process and call transform(x)."""
    ns: dict[str, Any] = {}
    exec(compile(source, "<transform_source>", "exec"), ns)  # noqa: S102 — trusted only
    if "transform" not in ns or not callable(ns["transform"]):
        raise ExecutionError("source does not define a callable transform(x)")
    return ns["transform"](x)


def evaluate(transform: Transform, x: dict[str, Any]) -> dict[str, Any]:
    """Ground-truth evaluation of a transform (via the canonical graph path)."""
    return run_graph(transform.graph, x)
