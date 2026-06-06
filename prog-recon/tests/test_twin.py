"""Twin construction tests (CHECKPOINT 3).

Must assert topology isomorphism and equal complexity tuples.
"""

from progrecon.config import load_config
from progrecon.corpus import generator, twin
from progrecon.dsl import graph

CFG = load_config()


def _base(seed=7, n=3, m=2):
    return generator.sample_transform(CFG.generators[1], n=n, m=m, target_depth=3, seed=seed, transform_id="A-sem")


def test_twin_preserves_topology_and_complexity():
    base = _base()
    tw = twin.make_abstract(base, seed=99, twin_id="A-sem~twin")
    # equal complexity tuple
    assert tw.complexity == base.complexity
    # topology isomorphic: identical node ids, edges, output_map
    assert [n.node_id for n in tw.graph.nodes] == [n.node_id for n in base.graph.nodes]
    assert {n.node_id: n.inputs for n in tw.graph.nodes} == {n.node_id: n.inputs for n in base.graph.nodes}
    assert tw.graph.output_map == base.graph.output_map
    # valid + abstract
    assert graph.validate(tw.graph, tw.schema) == []
    assert tw.semantic is False and tw.domain is None
    assert tw.twin_id == base.id


def test_twin_masks_field_names():
    base = _base()
    tw = twin.make_abstract(base, seed=1)
    assert [f.name for f in tw.schema.inputs] == [f"in_{i}" for i in range(tw.schema.n)]
    assert [f.name for f in tw.schema.outputs] == [f"out_{i}" for i in range(tw.schema.m)]
    assert all(f.semantic_label is None for f in tw.schema.inputs)


def test_twin_swaps_at_least_one_op_when_possible():
    # Across several seeds, a transform with neutral tier-0/1/2 ops should see
    # at least one op substituted in some twin.
    base = _base(seed=12, n=4, m=1)
    swapped_any = False
    for s in range(5):
        tw = twin.make_abstract(base, seed=s)
        if any(a.op != b.op for a, b in zip(base.graph.nodes, tw.graph.nodes, strict=True)):
            swapped_any = True
            break
    assert swapped_any


def test_pseudo_semantic_attaches_misleading_labels():
    base = _base()
    abstract = twin.make_abstract(base, seed=3)
    pseudo = twin.make_pseudo_semantic(abstract, seed=5)
    assert pseudo.semantic is True and pseudo.domain == "pseudo"
    assert all(f.semantic_label is not None for f in pseudo.schema.inputs)
    # structure unchanged vs the abstract source it dresses up
    assert pseudo.graph.output_map == abstract.graph.output_map
