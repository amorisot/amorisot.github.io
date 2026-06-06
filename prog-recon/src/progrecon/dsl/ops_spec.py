"""The OP SEMANTICS TABLE, as data — the single source of truth for op meaning.

Each op declares: tier, exact natural-language semantics (incl. pinned edge
cases), arity, the *type class* it accepts on value inputs, and >= 3
hand-checked test vectors. ``tests/test_ops.py`` executes every vector against
``ops.OPS`` and a human reviews the ``doc`` strings at CHECKPOINT 1.

This module also owns the small wiring type-system used by the generator and
validator: ``literal_type``, ``class_compatible``, ``op_accepts`` and
``infer_out_type``. Keeping these next to the table guarantees that semantics
and type-wiring never drift apart.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

# Concrete value types that flow through the dataflow graph.
ConcreteType = Literal["int", "float", "bool", "categorical"]

# Type *classes* an op may require on a value input.
# "num" = int or float; "any" = anything; "mixed" = op validates inputs itself.
InClass = Literal["int", "num", "bool", "categorical", "any", "mixed"]


class OpVector(BaseModel):
    """One hand-checked input/output example for an op."""

    model_config = ConfigDict(extra="forbid")
    params: dict[str, Any]
    args: list[Any]
    expected: Any


class OpSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    tier: int
    doc: str  # exact semantics + pinned edge cases
    variadic: bool
    min_arity: int
    max_arity: int  # for variadic ops, the upper bound the generator may wire
    in_class: InClass
    out_doc: str  # human note on the output type
    vectors: list[OpVector]


# ---------------------------------------------------------------------------
# THE TABLE
# ---------------------------------------------------------------------------

OPS_SPEC: dict[str, OpSpec] = {
    "mod_k": OpSpec(
        name="mod_k",
        tier=0,
        doc=(
            "x mod k. Result sign follows Python `%` (non-negative for positive k, "
            "so -3 mod 5 = 2). DECISION: integers only — float inputs are rejected "
            "at generation, never wired here. k is a positive integer constant."
        ),
        variadic=False,
        min_arity=1,
        max_arity=1,
        in_class="int",
        out_doc="int",
        vectors=[
            OpVector(params={"k": 3}, args=[7], expected=1),
            OpVector(params={"k": 5}, args=[-3], expected=2),
            OpVector(params={"k": 4}, args=[8], expected=0),
            OpVector(params={"k": 7}, args=[7], expected=0),
        ],
    ),
    "round_to_k": OpSpec(
        name="round_to_k",
        tier=0,
        doc=(
            "Round x to the nearest multiple of k. DECISION: half-way cases round "
            "HALF-TO-EVEN (banker's rounding, via Python round): round_to_k(5,k=10)=0, "
            "round_to_k(15,k=10)=20, round_to_k(25,k=10)=20. Output is int iff both x "
            "and k are int, else float."
        ),
        variadic=False,
        min_arity=1,
        max_arity=1,
        in_class="num",
        out_doc="int if input int (with int k) else float",
        vectors=[
            OpVector(params={"k": 10}, args=[5], expected=0),
            OpVector(params={"k": 10}, args=[15], expected=20),
            OpVector(params={"k": 10}, args=[25], expected=20),
            OpVector(params={"k": 10}, args=[12], expected=10),
            OpVector(params={"k": 5}, args=[3], expected=5),
        ],
    ),
    "digit_reverse": OpSpec(
        name="digit_reverse",
        tier=0,
        doc=(
            "Operate on abs(int(x)), reverse the decimal digits, DROP leading zeros, "
            "then reapply the original sign. digit_reverse(100)=1, "
            "digit_reverse(-120)=-21, digit_reverse(0)=0."
        ),
        variadic=False,
        min_arity=1,
        max_arity=1,
        in_class="int",
        out_doc="int",
        vectors=[
            OpVector(params={}, args=[100], expected=1),
            OpVector(params={}, args=[-120], expected=-21),
            OpVector(params={}, args=[1234], expected=4321),
            OpVector(params={}, args=[0], expected=0),
        ],
    ),
    "clip": OpSpec(
        name="clip",
        tier=0,
        doc="min(max(x, lo), hi). Requires lo <= hi (checked at generation).",
        variadic=False,
        min_arity=1,
        max_arity=1,
        in_class="num",
        out_doc="int if x and bounds int else float",
        vectors=[
            OpVector(params={"lo": 0, "hi": 10}, args=[15], expected=10),
            OpVector(params={"lo": 0, "hi": 10}, args=[-5], expected=0),
            OpVector(params={"lo": 0, "hi": 10}, args=[7], expected=7),
        ],
    ),
    "affine": OpSpec(
        name="affine",
        tier=0,
        doc="a*x + b with constant a, b. Neutral tier-0 op (used for twins).",
        variadic=False,
        min_arity=1,
        max_arity=1,
        in_class="num",
        out_doc="int if x,a,b all int else float",
        vectors=[
            OpVector(params={"a": 2, "b": 3}, args=[4], expected=11),
            OpVector(params={"a": 0, "b": 5}, args=[100], expected=5),
            OpVector(params={"a": -1, "b": 0}, args=[4], expected=-4),
        ],
    ),
    "abs_val": OpSpec(
        name="abs_val",
        tier=0,
        doc="abs(x). Neutral tier-0 op.",
        variadic=False,
        min_arity=1,
        max_arity=1,
        in_class="num",
        out_doc="same numeric type as input",
        vectors=[
            OpVector(params={}, args=[-3], expected=3),
            OpVector(params={}, args=[3], expected=3),
            OpVector(params={}, args=[0], expected=0),
        ],
    ),
    "negate": OpSpec(
        name="negate",
        tier=0,
        doc="-x. Neutral tier-0 op.",
        variadic=False,
        min_arity=1,
        max_arity=1,
        in_class="num",
        out_doc="same numeric type as input",
        vectors=[
            OpVector(params={}, args=[5], expected=-5),
            OpVector(params={}, args=[-2], expected=2),
            OpVector(params={}, args=[0], expected=0),
        ],
    ),
    "weighted_sum": OpSpec(
        name="weighted_sum",
        tier=1,
        doc=(
            "sum_i w_i * in_i with fixed constant weights w (len(w) == arity). "
            "Computed in float64; ALWAYS returns float."
        ),
        variadic=True,
        min_arity=2,
        max_arity=6,
        in_class="num",
        out_doc="float (always)",
        vectors=[
            OpVector(params={"w": [1, 1]}, args=[2, 3], expected=5.0),
            OpVector(params={"w": [2, 0.5]}, args=[3, 4], expected=8.0),
            OpVector(params={"w": [1, -1]}, args=[5, 2], expected=3.0),
        ],
    ),
    "sum_inputs": OpSpec(
        name="sum_inputs",
        tier=1,
        doc="Sum of all inputs. Neutral tier-1 op.",
        variadic=True,
        min_arity=2,
        max_arity=6,
        in_class="num",
        out_doc="int if all inputs int else float",
        vectors=[
            OpVector(params={}, args=[1, 2, 3], expected=6),
            OpVector(params={}, args=[5, 0], expected=5),
            OpVector(params={}, args=[-1, 1], expected=0),
        ],
    ),
    "product": OpSpec(
        name="product",
        tier=1,
        doc="Product of all inputs. Neutral tier-1 op.",
        variadic=True,
        min_arity=2,
        max_arity=4,
        in_class="num",
        out_doc="int if all inputs int else float",
        vectors=[
            OpVector(params={}, args=[2, 3, 4], expected=24),
            OpVector(params={}, args=[5, 0], expected=0),
            OpVector(params={}, args=[-2, 3], expected=-6),
        ],
    ),
    "diff": OpSpec(
        name="diff",
        tier=1,
        doc="a - b (first input minus second). Neutral tier-1 op.",
        variadic=False,
        min_arity=2,
        max_arity=2,
        in_class="num",
        out_doc="int if both int else float",
        vectors=[
            OpVector(params={}, args=[5, 3], expected=2),
            OpVector(params={}, args=[3, 5], expected=-2),
            OpVector(params={}, args=[0, 0], expected=0),
        ],
    ),
    "select_if": OpSpec(
        name="select_if",
        tier=1,
        doc=(
            "Inputs (cond, a, b). Returns a if cond is truthy else b. Truthiness: "
            "bool -> itself; numeric -> (x != 0); categorical -> membership in the "
            "params['truthy'] set (absent/empty => falsy). a and b must share a type."
        ),
        variadic=False,
        min_arity=3,
        max_arity=3,
        in_class="mixed",
        out_doc="type of the a/b branches (which must match)",
        vectors=[
            OpVector(params={}, args=[1, 10, 20], expected=10),
            OpVector(params={}, args=[0, 10, 20], expected=20),
            OpVector(params={}, args=[True, "a", "b"], expected="a"),
            OpVector(params={"truthy": ["yes"]}, args=["yes", 1, 2], expected=1),
            OpVector(params={"truthy": ["yes"]}, args=["no", 1, 2], expected=2),
        ],
    ),
    "argmax_index": OpSpec(
        name="argmax_index",
        tier=2,
        doc="Index of the maximum over the inputs. DECISION: ties -> LOWEST index.",
        variadic=True,
        min_arity=2,
        max_arity=6,
        in_class="num",
        out_doc="int (an index)",
        vectors=[
            OpVector(params={}, args=[1, 5, 3], expected=1),
            OpVector(params={}, args=[5, 5, 1], expected=0),
            OpVector(params={}, args=[1, 2, 3, 3], expected=2),
        ],
    ),
    "count_above": OpSpec(
        name="count_above",
        tier=2,
        doc="Count of inputs STRICTLY greater than thr (>, not >=).",
        variadic=True,
        min_arity=2,
        max_arity=6,
        in_class="num",
        out_doc="int (a count)",
        vectors=[
            OpVector(params={"thr": 5}, args=[3, 5, 7, 9], expected=2),
            OpVector(params={"thr": 0}, args=[-1, 0, 1], expected=1),
            OpVector(params={"thr": 10}, args=[1, 2, 3], expected=0),
        ],
    ),
    "median": OpSpec(
        name="median",
        tier=2,
        doc=(
            "Median of the inputs. DECISION: even length -> mean of the two middle "
            "values (so output may be non-integer). ALWAYS returns float."
        ),
        variadic=True,
        min_arity=2,
        max_arity=6,
        in_class="num",
        out_doc="float (always)",
        vectors=[
            OpVector(params={}, args=[1, 2, 3], expected=2.0),
            OpVector(params={}, args=[1, 2, 3, 4], expected=2.5),
            OpVector(params={}, args=[3, 1, 2], expected=2.0),
        ],
    ),
    "max_val": OpSpec(
        name="max_val",
        tier=2,
        doc="Maximum of the inputs. Neutral tier-2 op.",
        variadic=True,
        min_arity=2,
        max_arity=6,
        in_class="num",
        out_doc="float if any input float else int",
        vectors=[
            OpVector(params={}, args=[1, 5, 3], expected=5),
            OpVector(params={}, args=[-1, -5], expected=-1),
            OpVector(params={}, args=[2, 2, 2], expected=2),
        ],
    ),
    "min_val": OpSpec(
        name="min_val",
        tier=2,
        doc="Minimum of the inputs. Neutral tier-2 op.",
        variadic=True,
        min_arity=2,
        max_arity=6,
        in_class="num",
        out_doc="float if any input float else int",
        vectors=[
            OpVector(params={}, args=[1, 5, 3], expected=1),
            OpVector(params={}, args=[-1, -5], expected=-5),
            OpVector(params={}, args=[2, 2, 2], expected=2),
        ],
    ),
    "bracket_dispatch": OpSpec(
        name="bracket_dispatch",
        tier=3,
        doc=(
            "Piecewise dispatch on x. thresholds t1<t2<... ascending; values has "
            "len(thresholds)+1 entries. Intervals are HALF-OPEN [t_i, t_{i+1}); a "
            "value below t1 maps to the first bracket (values[0]); x>=t_last maps to "
            "values[-1]."
        ),
        variadic=False,
        min_arity=1,
        max_arity=1,
        in_class="num",
        out_doc="type of the bracket values",
        vectors=[
            OpVector(params={"thresholds": [10, 20], "values": [0, 1, 2]}, args=[5], expected=0),
            OpVector(params={"thresholds": [10, 20], "values": [0, 1, 2]}, args=[10], expected=1),
            OpVector(params={"thresholds": [10, 20], "values": [0, 1, 2]}, args=[15], expected=1),
            OpVector(params={"thresholds": [10, 20], "values": [0, 1, 2]}, args=[20], expected=2),
            OpVector(params={"thresholds": [10, 20], "values": [0, 1, 2]}, args=[25], expected=2),
        ],
    ),
    "lookup_table": OpSpec(
        name="lookup_table",
        tier=3,
        doc=(
            "Categorical key -> value via a fixed table. DECISION: an unseen key MUST "
            "NOT occur — P_x guarantees coverage and the identifiability check enforces "
            "it; if it ever happens the executor raises KeyError (never silently maps)."
        ),
        variadic=False,
        min_arity=1,
        max_arity=1,
        in_class="categorical",
        out_doc="type of the table values",
        vectors=[
            OpVector(params={"table": {"a": 1, "b": 2}}, args=["a"], expected=1),
            OpVector(params={"table": {"a": 1, "b": 2}}, args=["b"], expected=2),
            OpVector(params={"table": {"x": 10}}, args=["x"], expected=10),
        ],
    ),
}


# ---------------------------------------------------------------------------
# Wiring type-system helpers (shared by generator + validator + twins)
# ---------------------------------------------------------------------------


def literal_type(v: Any) -> ConcreteType:
    """Concrete type of a Python literal (bool checked before int)."""
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "categorical"
    raise TypeError(f"unsupported literal type: {type(v).__name__}")


def class_compatible(required: InClass, concrete: ConcreteType) -> bool:
    if required == "any":
        return True
    if required == "num":
        return concrete in ("int", "float")
    if required == "int":
        return concrete == "int"
    if required == "bool":
        return concrete == "bool"
    if required == "categorical":
        return concrete == "categorical"
    # "mixed" => op validates its own inputs (e.g. select_if)
    return True


def op_accepts(op: str, in_types: list[ConcreteType]) -> bool:
    """Whether an op can be wired to inputs of the given concrete types."""
    spec = OPS_SPEC[op]
    arity = len(in_types)
    if spec.variadic:
        if not (spec.min_arity <= arity <= spec.max_arity):
            return False
    else:
        if arity != spec.min_arity:
            return False
    if op == "select_if":
        # (cond, a, b): cond any scalar; a and b must share a type.
        return in_types[1] == in_types[2]
    return all(class_compatible(spec.in_class, t) for t in in_types)


def _homogeneous_literal_type(values: list[Any]) -> ConcreteType:
    types = {literal_type(v) for v in values}
    if "categorical" in types:
        return "categorical"
    if "float" in types:
        return "float"
    if types == {"bool"}:
        return "bool"
    return "int"


def infer_out_type(
    op: str, in_types: list[ConcreteType], params: dict[str, Any]
) -> ConcreteType:
    """The concrete output type of an op given its input types and params."""
    if op in ("mod_k", "digit_reverse", "argmax_index", "count_above"):
        return "int"
    if op in ("weighted_sum", "median"):
        return "float"
    if op == "round_to_k":
        return "int" if in_types[0] == "int" and isinstance(params.get("k"), int) else "float"
    if op == "clip":
        bounds_int = isinstance(params.get("lo"), int) and isinstance(params.get("hi"), int)
        bounds_int = bounds_int and not isinstance(params.get("lo"), bool)
        return "int" if in_types[0] == "int" and bounds_int else "float"
    if op == "affine":
        a, b = params.get("a"), params.get("b")
        a_int = isinstance(a, int) and not isinstance(a, bool)
        b_int = isinstance(b, int) and not isinstance(b, bool)
        return "int" if in_types[0] == "int" and a_int and b_int else "float"
    if op in ("abs_val", "negate"):
        return in_types[0]
    if op in ("sum_inputs", "product", "diff", "max_val", "min_val"):
        return "int" if all(t == "int" for t in in_types) else "float"
    if op == "select_if":
        return in_types[1]
    if op == "bracket_dispatch":
        return _homogeneous_literal_type(list(params["values"]))
    if op == "lookup_table":
        return _homogeneous_literal_type(list(params["table"].values()))
    raise KeyError(f"no out-type rule for op {op!r}")


def ops_by_tier(tiers: list[int]) -> list[str]:
    return [name for name, s in OPS_SPEC.items() if s.tier in tiers]


def render_op_table_markdown() -> str:
    """Render the semantics table to Markdown (for docs/op_semantics.md)."""
    lines = [
        "# Op semantics table",
        "",
        "> Generated from `src/progrecon/dsl/ops_spec.py` — the single source of "
        "truth. Regenerate with `uv run progrecon dump-ops`. **Review every row "
        "and edge case at CHECKPOINT 1.**",
        "",
    ]
    for tier in sorted({s.tier for s in OPS_SPEC.values()}):
        lines.append(f"## Tier {tier}")
        lines.append("")
        lines.append("| op | arity | in-class | out | semantics & pinned edge cases |")
        lines.append("|---|---|---|---|---|")
        for name in sorted(OPS_SPEC):
            s = OPS_SPEC[name]
            if s.tier != tier:
                continue
            arity = (
                f"{s.min_arity}..{s.max_arity}" if s.variadic else str(s.min_arity)
            )
            doc = s.doc.replace("|", r"\|")
            lines.append(f"| `{name}` | {arity} | {s.in_class} | {s.out_doc} | {doc} |")
        lines.append("")
    lines.append("## Hand-checked vectors")
    lines.append("")
    for name in sorted(OPS_SPEC):
        s = OPS_SPEC[name]
        lines.append(f"- `{name}`:")
        for v in s.vectors:
            lines.append(f"  - `{name}({v.params}, *{v.args})` = `{v.expected!r}`")
    lines.append("")
    return "\n".join(lines)
