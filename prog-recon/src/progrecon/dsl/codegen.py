"""Codegen: DataflowGraph -> self-contained ground-truth Python source.

``Transform.source`` is ALWAYS produced here from the graph; it is never
hand-maintained. The generated source is self-contained (only ``import math``)
so it can be displayed in prompts, used for `infill` mode, and executed in the
sandbox without importing this package.

Single source of truth for op behavior: the op function bodies are extracted
verbatim from ``dsl.ops`` via ``inspect.getsource`` and embedded as a prelude,
so the codegen'd source and the canonical executor can never disagree.
"""

from __future__ import annotations

import inspect
import textwrap

from ..types import DataflowGraph, Schema
from . import ops
from .graph import topological_order


def _var(node_id: str) -> str:
    """A safe local variable name for a node (cannot clash with op names)."""
    safe = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in node_id)
    return f"_n_{safe}"


def _op_prelude(used_ops: set[str]) -> str:
    parts = ["import math", ""]
    for name in sorted(used_ops):
        fn = ops.OPS[name]
        src = textwrap.dedent(inspect.getsource(fn))
        parts.append(src.rstrip())
        parts.append("")
    return "\n".join(parts)


def graph_to_source(g: DataflowGraph, schema: Schema) -> str:
    """Generate `def transform(x: dict) -> dict` for the graph."""
    order = topological_order(g)
    field_names = {f.name for f in schema.inputs}
    used_ops = {g.node(nid).op for nid in order}

    body_lines: list[str] = []
    for nid in order:
        nd = g.node(nid)
        arg_exprs = []
        for inp in nd.inputs:
            if inp in field_names:
                arg_exprs.append(f"x[{inp!r}]")
            else:
                arg_exprs.append(_var(inp))
        args_joined = ", ".join([repr(nd.params)] + arg_exprs)
        body_lines.append(f"    {_var(nid)} = {nd.op}({args_joined})")

    ret_items = ", ".join(
        f"{out_name!r}: {_var(producer)}" for out_name, producer in g.output_map.items()
    )
    body_lines.append(f"    return {{{ret_items}}}")

    prelude = _op_prelude(used_ops)
    fn_src = "def transform(x):\n" + "\n".join(body_lines) + "\n"
    return prelude + "\n" + fn_src
