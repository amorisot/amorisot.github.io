"""Hand-checked op semantics tests (CHECKPOINT 1 — the most critical gate).

A bug here silently corrupts the entire dataset, so every op's pinned edge
cases are exercised both via the declarative vectors in ``ops_spec`` and via
extra explicit cases below.
"""

import math

import pytest

from progrecon.dsl import ops
from progrecon.dsl.ops_spec import (
    OPS_SPEC,
    infer_out_type,
    literal_type,
    op_accepts,
)


def _eq(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def test_registry_matches_spec():
    # Every op has a spec and an implementation; sets are identical.
    assert set(ops.OPS.keys()) == set(OPS_SPEC.keys())


def test_every_op_has_three_vectors_and_docs():
    for name, spec in OPS_SPEC.items():
        assert len(spec.vectors) >= 3, f"{name} needs >= 3 vectors"
        assert spec.doc.strip(), f"{name} missing doc"
        assert spec.out_doc.strip(), f"{name} missing out_doc"


@pytest.mark.parametrize("name", sorted(OPS_SPEC.keys()))
def test_declared_vectors(name):
    spec = OPS_SPEC[name]
    fn = ops.OPS[name]
    for v in spec.vectors:
        got = fn(v.params, *v.args)
        assert _eq(got, v.expected), f"{name}{tuple(v.args)} -> {got!r}, expected {v.expected!r}"


# --- Extra explicit edge cases (belt-and-suspenders) -----------------------


def test_mod_k_negative_follows_python():
    assert ops.mod_k({"k": 5}, -3) == 2
    assert ops.mod_k({"k": 5}, -1) == 4
    assert ops.mod_k({"k": 5}, 0) == 0


def test_round_to_k_half_to_even():
    # 5 and 25 are halfway -> round to even multiple (0 and 20); 15 -> 20.
    assert ops.round_to_k({"k": 10}, 5) == 0
    assert ops.round_to_k({"k": 10}, 15) == 20
    assert ops.round_to_k({"k": 10}, 25) == 20
    assert ops.round_to_k({"k": 10}, 35) == 40
    # int in -> int out
    assert isinstance(ops.round_to_k({"k": 10}, 12), int)
    # float in -> float out
    assert isinstance(ops.round_to_k({"k": 10}, 12.0), float)


def test_digit_reverse_pinned():
    assert ops.digit_reverse({}, 100) == 1
    assert ops.digit_reverse({}, -120) == -21
    assert ops.digit_reverse({}, 10) == 1
    assert ops.digit_reverse({}, 0) == 0


def test_argmax_ties_lowest_index():
    assert ops.argmax_index({}, 5, 5, 5) == 0
    assert ops.argmax_index({}, 1, 9, 9) == 1


def test_count_above_strict():
    assert ops.count_above({"thr": 5}, 5, 5, 5) == 0  # not >=
    assert ops.count_above({"thr": 5}, 6, 5, 7) == 2


def test_median_even_is_mean_of_middle():
    assert ops.median({}, 1, 2, 3, 4) == 2.5
    assert ops.median({}, 10, 20) == 15.0
    assert isinstance(ops.median({}, 1, 2, 3), float)  # always float


def test_bracket_dispatch_half_open():
    p = {"thresholds": [10, 20], "values": ["lo", "mid", "hi"]}
    assert ops.bracket_dispatch(p, 9) == "lo"
    assert ops.bracket_dispatch(p, 10) == "mid"  # [10,20) inclusive lower
    assert ops.bracket_dispatch(p, 19) == "mid"
    assert ops.bracket_dispatch(p, 20) == "hi"
    assert ops.bracket_dispatch(p, 1000) == "hi"


def test_lookup_table_raises_on_unseen():
    with pytest.raises(KeyError):
        ops.lookup_table({"table": {"a": 1}}, "z")


def test_select_if_truthiness():
    assert ops.select_if({}, 0.0, "yes", "no") == "no"
    assert ops.select_if({}, 0.1, "yes", "no") == "yes"
    assert ops.select_if({}, False, 1, 2) == 2
    assert ops.select_if({"truthy": ["A", "B"]}, "B", 1, 2) == 1
    assert ops.select_if({"truthy": ["A", "B"]}, "C", 1, 2) == 2


# --- Branch helpers --------------------------------------------------------


def test_branch_arms_and_taken():
    assert ops.branch_arms("select_if", {}) == 2
    assert ops.branch_arms("bracket_dispatch", {"values": [0, 1, 2]}) == 3
    assert ops.branch_arms("mod_k", {"k": 3}) == 0
    assert ops.branch_arm_taken("select_if", {}, 1, "a", "b") == 0
    assert ops.branch_arm_taken("select_if", {}, 0, "a", "b") == 1
    p = {"thresholds": [10, 20], "values": [0, 1, 2]}
    assert ops.branch_arm_taken("bracket_dispatch", p, 5) == 0
    assert ops.branch_arm_taken("bracket_dispatch", p, 25) == 2


# --- Type-system helpers ---------------------------------------------------


def test_literal_type_bool_before_int():
    assert literal_type(True) == "bool"
    assert literal_type(3) == "int"
    assert literal_type(3.0) == "float"
    assert literal_type("x") == "categorical"


def test_op_accepts_arity_and_class():
    assert op_accepts("mod_k", ["int"])
    assert not op_accepts("mod_k", ["float"])  # int-only
    assert not op_accepts("mod_k", ["int", "int"])  # arity 1
    assert op_accepts("weighted_sum", ["int", "float"])  # variadic num
    assert not op_accepts("weighted_sum", ["int"])  # below min arity
    assert op_accepts("select_if", ["int", "float", "float"])  # a,b match
    assert not op_accepts("select_if", ["int", "int", "float"])  # a,b differ
    assert op_accepts("lookup_table", ["categorical"])
    assert not op_accepts("lookup_table", ["int"])


def test_infer_out_type():
    assert infer_out_type("mod_k", ["int"], {"k": 3}) == "int"
    assert infer_out_type("round_to_k", ["int"], {"k": 10}) == "int"
    assert infer_out_type("round_to_k", ["float"], {"k": 10}) == "float"
    assert infer_out_type("weighted_sum", ["int", "int"], {"w": [1, 1]}) == "float"
    assert infer_out_type("sum_inputs", ["int", "int"], {}) == "int"
    assert infer_out_type("sum_inputs", ["int", "float"], {}) == "float"
    assert infer_out_type("median", ["int", "int"], {}) == "float"
    assert infer_out_type("affine", ["int"], {"a": 2, "b": 3}) == "int"
    assert infer_out_type("affine", ["int"], {"a": 2.0, "b": 3}) == "float"
    assert infer_out_type("bracket_dispatch", ["int"], {"thresholds": [1], "values": ["a", "b"]}) == "categorical"
    assert infer_out_type("lookup_table", ["categorical"], {"table": {"a": 1}}) == "int"
