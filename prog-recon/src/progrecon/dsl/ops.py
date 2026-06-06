"""Op implementations — pure functions, one per DSL op.

Every op has the uniform signature ``op(params: dict, *args) -> value``:
* ``params`` holds the op's *constants* (e.g. ``{"k": 8}``).
* ``*args`` are the *value inputs* (results of upstream nodes / input fields),
  in argument order.

CRITICAL invariants (so codegen can extract these by source text):
* Each function is module-level and self-contained: it uses only builtins and
  ``math`` (which the codegen prelude also imports). No closures, no globals,
  no references to other functions in this module.
* No randomness, no I/O, no mutation of inputs.

Edge-case semantics are pinned in ``ops_spec.py`` (the semantics table) and
verified by ``tests/test_ops.py``. Do not change behavior here without updating
the table and its hand-checked vectors.
"""

from __future__ import annotations

import math  # noqa: F401 — available to ops and to the codegen prelude
from collections.abc import Callable
from typing import Any

# ---------------------------------------------------------------------------
# Tier 0 — scalar -> scalar (arity 1)
# ---------------------------------------------------------------------------


def mod_k(params, *args):
    # x mod k, Python `%` semantics (non-negative result for positive k).
    # Integers only (enforced at generation; floats are never wired here).
    return args[0] % params["k"]


def round_to_k(params, *args):
    # Round to the nearest multiple of k; ties round half-to-even (Python round).
    k = params["k"]
    x = args[0]
    r = k * round(x / k)
    if isinstance(x, int) and not isinstance(x, bool) and isinstance(k, int):
        return int(r)
    return float(r)


def digit_reverse(params, *args):
    # Reverse the decimal digits of abs(int(x)), drop leading zeros, reapply sign.
    x = args[0]
    sign = -1 if x < 0 else 1
    rev = int(str(abs(int(x)))[::-1])
    return sign * rev


def clip(params, *args):
    # min(max(x, lo), hi); requires lo <= hi (checked at generation).
    return min(max(args[0], params["lo"]), params["hi"])


def affine(params, *args):
    # a*x + b. Neutral tier-0 op (handy for twins).
    return params["a"] * args[0] + params["b"]


def abs_val(params, *args):
    return abs(args[0])


def negate(params, *args):
    return -args[0]


# ---------------------------------------------------------------------------
# Tier 1 — combine a few inputs
# ---------------------------------------------------------------------------


def weighted_sum(params, *args):
    # sum_i w_i * in_i. Weights are fixed constants; float64 precision.
    w = params["w"]
    total = 0.0
    for i in range(len(args)):
        total += w[i] * args[i]
    return total


def sum_inputs(params, *args):
    total = 0
    for a in args:
        total += a
    return total


def product(params, *args):
    p = 1
    for a in args:
        p *= a
    return p


def diff(params, *args):
    return args[0] - args[1]


def select_if(params, *args):
    # a if cond truthy else b. Truthiness: bool -> itself; numeric -> (x != 0);
    # categorical -> membership in params["truthy"] (default empty -> falsy).
    cond = args[0]
    a = args[1]
    b = args[2]
    if isinstance(cond, bool):
        truthy = cond
    elif isinstance(cond, (int, float)):
        truthy = cond != 0
    else:
        truthy = cond in params.get("truthy", [])
    return a if truthy else b


# ---------------------------------------------------------------------------
# Tier 2 — reductions / order statistics over a slice
# ---------------------------------------------------------------------------


def argmax_index(params, *args):
    # Index of the max over the inputs; ties resolve to the LOWEST index.
    best = 0
    for i in range(1, len(args)):
        if args[i] > args[best]:
            best = i
    return best


def count_above(params, *args):
    # Count of inputs STRICTLY greater than thr (not >=).
    thr = params["thr"]
    c = 0
    for a in args:
        if a > thr:
            c += 1
    return c


def median(params, *args):
    # Median; even length -> mean of the two middle values (may be non-integer).
    # Always returns float so the output type is unambiguous.
    s = sorted(args)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


def max_val(params, *args):
    return max(args)


def min_val(params, *args):
    return min(args)


# ---------------------------------------------------------------------------
# Tier 3 — dispatch / lookup
# ---------------------------------------------------------------------------


def bracket_dispatch(params, *args):
    # Half-open brackets [t_i, t_{i+1}); value below t1 -> first bracket.
    # thresholds ascending (t1<t2<...); values has len(thresholds)+1 entries.
    x = args[0]
    thresholds = params["thresholds"]
    values = params["values"]
    idx = 0
    for t in thresholds:
        if x >= t:
            idx += 1
        else:
            break
    return values[idx]


def lookup_table(params, *args):
    # Categorical key -> value. Unseen key must never occur (P_x guarantees
    # coverage; identifiability enforces it). Raise loudly if it does.
    table = params["table"]
    key = args[0]
    if key not in table:
        raise KeyError("lookup_table: unseen key " + repr(key))
    return table[key]


# ---------------------------------------------------------------------------
# Registry + branch helpers
# ---------------------------------------------------------------------------

OPS: dict[str, Callable[..., Any]] = {
    "mod_k": mod_k,
    "round_to_k": round_to_k,
    "digit_reverse": digit_reverse,
    "clip": clip,
    "affine": affine,
    "abs_val": abs_val,
    "negate": negate,
    "weighted_sum": weighted_sum,
    "sum_inputs": sum_inputs,
    "product": product,
    "diff": diff,
    "select_if": select_if,
    "argmax_index": argmax_index,
    "count_above": count_above,
    "median": median,
    "max_val": max_val,
    "min_val": min_val,
    "bracket_dispatch": bracket_dispatch,
    "lookup_table": lookup_table,
}

# Ops that introduce conditional branches (used for complexity + coverage).
BRANCH_OPS = frozenset({"select_if", "bracket_dispatch"})


def get_op(name: str) -> Callable[..., Any]:
    if name not in OPS:
        raise KeyError(f"unknown op {name!r}")
    return OPS[name]


def branch_arms(op: str, params: dict[str, Any]) -> int:
    """Number of distinct arms a branch op exposes (0 for non-branch ops)."""
    if op == "select_if":
        return 2
    if op == "bracket_dispatch":
        return len(params["values"])
    return 0


def branch_arm_taken(op: str, params: dict[str, Any], *args: Any) -> int:
    """Which arm index fired given the node's input values (for coverage sim)."""
    if op == "select_if":
        cond = args[0]
        if isinstance(cond, bool):
            truthy = cond
        elif isinstance(cond, (int, float)):
            truthy = cond != 0
        else:
            truthy = cond in params.get("truthy", [])
        return 0 if truthy else 1
    if op == "bracket_dispatch":
        x = args[0]
        idx = 0
        for t in params["thresholds"]:
            if x >= t:
                idx += 1
            else:
                break
        return idx
    return 0
