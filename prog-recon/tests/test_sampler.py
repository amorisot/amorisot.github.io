"""Sampler tests (CHECKPOINT 4) — incl. the D_test firewall."""

from progrecon.config import load_config
from progrecon.data import sampler
from progrecon.dsl import executor
from progrecon.types import DataflowGraph, FieldSpec, OpNode, Schema

from .conftest import build_transform

CFG = load_config()


def _int_transform(lo=0, hi=10):
    schema = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=lo, high=hi)],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="mod_k", params={"k": 3}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    return build_transform(schema, g, tid="B-int")


def _float_transform():
    schema = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=50),
                FieldSpec(name="in_1", type="int", low=0, high=50)],
        outputs=[FieldSpec(name="out_0", type="float")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="weighted_sum", params={"w": [1.0, 2.0]}, inputs=["in_0", "in_1"])],
        output_map={"out_0": "n0"},
    )
    return build_transform(schema, g, tid="B-float")


def test_firewall_disjoint_row_ids():
    t = _int_transform()
    cfg = CFG.model_copy(update={"scoring": CFG.scoring.model_copy(update={"test_id_size": 100, "test_ood_size": 100})})
    bundle = sampler.make_dataset(t, k=50, seed=1, cfg=cfg)
    assert bundle.firewall_ok()
    assert all(r < 10_000_000 for r in bundle.train.row_ids)
    assert all(10_000_000 <= r < 20_000_000 for r in bundle.test_id.row_ids)
    assert all(r >= 20_000_000 for r in bundle.test_ood.row_ids)
    assert bundle.train.row_ids.isdisjoint(bundle.test_id.row_ids)


def test_train_and_test_seeds_independent():
    t = _int_transform()
    tr = sampler.make_samples(t, 20, 5, "train")
    ti = sampler.make_samples(t, 20, 5, "test_id")
    assert tr.seed != ti.seed


def test_ground_truth_correct():
    t = _float_transform()
    ss = sampler.make_samples(t, 100, 3, "test_id")
    for ex in ss.examples:
        assert ex.y == executor.run_graph(t.graph, ex.x)


def test_ood_widens_numeric_range():
    t = _int_transform(lo=0, hi=10)
    ss = sampler.make_samples(t, 1000, 9, "test_ood")
    vals = [ex.x["in_0"] for ex in ss.examples]
    assert min(vals) < 0 or max(vals) > 10  # OOD draws escape the training support


def test_eps_noise_train_only():
    t = _float_transform()
    cfg = CFG
    train = sampler.make_samples(t, 200, 4, "train", noise="eps", cfg=cfg)
    # at least some train labels differ from exact ground truth
    diffs = sum(1 for ex in train.examples if ex.y["out_0"] != executor.run_graph(t.graph, ex.x)["out_0"])
    assert diffs > 0
    # test split is never noised
    test = sampler.make_samples(t, 200, 4, "test_id", cfg=cfg)
    assert all(ex.y == executor.run_graph(t.graph, ex.x) for ex in test.examples)


def test_flip_noise_categorical():
    schema = Schema(
        inputs=[FieldSpec(name="in_0", type="categorical", categories=["a", "b", "c"])],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="lookup_table", params={"table": {"a": 1, "b": 2, "c": 3}}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    # categorical OUTPUT so flip applies; use bracket-like via lookup with str values
    schema2 = Schema(
        inputs=[FieldSpec(name="in_0", type="categorical", categories=["a", "b", "c"])],
        outputs=[FieldSpec(name="out_0", type="categorical")],
    )
    g2 = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="lookup_table", params={"table": {"a": "x", "b": "y", "c": "z"}}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    t = build_transform(schema2, g2, tid="B-cat")
    _ = g, schema  # unused alt fixture
    train = sampler.make_samples(t, 300, 2, "train", noise="flip")
    flips = sum(1 for ex in train.examples if ex.y["out_0"] != executor.run_graph(t.graph, ex.x)["out_0"])
    assert flips > 0


def test_branch_bias_covers_all_arms():
    schema = Schema(
        inputs=[FieldSpec(name="in_0", type="int", low=0, high=100)],
        outputs=[FieldSpec(name="out_0", type="int")],
    )
    g = DataflowGraph(
        nodes=[OpNode(node_id="n0", op="bracket_dispatch", params={"thresholds": [33, 66], "values": [0, 1, 2]}, inputs=["in_0"])],
        output_map={"out_0": "n0"},
    )
    t = build_transform(schema, g, tid="B-branchy")
    train = sampler.make_samples(t, 6, 1, "train", branch_bias=True)
    from progrecon.dsl.ops import branch_arm_taken
    arms = set()
    for ex in train.examples:
        arms.add(branch_arm_taken("bracket_dispatch", {"thresholds": [33, 66], "values": [0, 1, 2]}, ex.x["in_0"]))
    assert arms == {0, 1, 2}
