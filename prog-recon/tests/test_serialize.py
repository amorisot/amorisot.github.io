"""Serialization tests (CHECKPOINT 4)."""

import pytest

from progrecon.data import sampler, serialize
from progrecon.types import DataflowGraph, FieldSpec, OpNode, Schema

from .conftest import build_transform


def _semantic_transform():
    schema = Schema(
        inputs=[
            FieldSpec(name="price", type="int", low=1, high=100, semantic_label="unit price (USD)"),
            FieldSpec(name="qty", type="int", low=1, high=10, semantic_label="quantity"),
        ],
        outputs=[FieldSpec(name="subtotal", type="float", semantic_label="line subtotal")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="weighted_sum", params={"w": [1.0, 1.0]}, inputs=["price", "qty"])],
        output_map={"subtotal": "n0"},
    )
    return build_transform(schema, g, tid="A-price", semantic=True)


def test_for_prompt_refuses_test_split():
    t = _semantic_transform()
    test = sampler.make_samples(t, 5, 1, "test_id")
    with pytest.raises(ValueError, match="never reach the model"):
        serialize.for_prompt(test, t.schema, mask_names=True)


def test_mask_names_hides_labels():
    t = _semantic_transform()
    train = sampler.make_samples(t, 5, 1, "train")
    shown = serialize.for_prompt(train, t.schema, mask_names=False)
    masked = serialize.for_prompt(train, t.schema, mask_names=True)
    assert "unit price (USD)" in shown
    assert "unit price (USD)" not in masked
    # the literal field-name keys are present in both (so candidates key correctly)
    assert "price" in masked and "price" in shown


def test_serialization_is_deterministic():
    t = _semantic_transform()
    train = sampler.make_samples(t, 8, 2, "train")
    a = serialize.for_prompt(train, t.schema, mask_names=True)
    b = serialize.for_prompt(train, t.schema, mask_names=True)
    assert a == b


def test_table_and_json_present():
    t = _semantic_transform()
    train = sampler.make_samples(t, 4, 3, "train")
    out = serialize.for_prompt(train, t.schema, mask_names=True)
    assert "Examples (JSON):" in out
    assert "→" in out  # table arrow
    assert out.count("\n") > 4
