"""Shared test helpers."""

from __future__ import annotations

from progrecon.dsl import codegen
from progrecon.dsl.graph import compute_complexity_tuple, validate_or_raise
from progrecon.types import DataflowGraph, Schema, Transform


def build_transform(
    schema: Schema,
    graph: DataflowGraph,
    *,
    tid: str = "TEST-1",
    corpus: str = "B",
    semantic: bool = False,
    px_seed: int = 0,
) -> Transform:
    """Construct a fully-populated Transform from a schema+graph (validates,
    codegen's source, computes complexity) — used to hand-build test fixtures."""
    validate_or_raise(graph, schema)
    return Transform(
        id=tid,
        corpus=corpus,  # type: ignore[arg-type]
        semantic=semantic,
        schema=schema,
        source=codegen.graph_to_source(graph, schema),
        graph=graph,
        complexity=compute_complexity_tuple(graph, schema),
        px_seed=px_seed,
    )
