"""Full-corpus assembly tests (Phase 7 step 2-3): 50 B + 50 A + 50 twins."""

import random
import tempfile
from pathlib import Path

import pytest

from progrecon.config import load_config
from progrecon.corpus import authored, build_corpus
from progrecon.data.sampler import draw_x
from progrecon.dsl import executor

CFG = load_config()


@pytest.fixture(scope="module")
def manifest():
    return build_corpus.build_full_corpus(seed=42, cfg=CFG)


def test_authored_all_identified():
    kept, quar = authored.load_corpus_a(n_probes=400)
    assert len(kept) == 50
    assert quar == [], [q[0] for q in quar]


def test_authored_source_matches_graph():
    rng = random.Random(0)
    for t in authored.build_all():
        for _ in range(20):
            x = draw_x(t.schema, rng)
            assert executor.run_graph(t.graph, x) == executor.run_source(t.source, x)


def test_corpus_counts(manifest):
    s = manifest.summary()
    assert s["corpus_a"] == 50
    assert s["corpus_b"] == 50
    assert s["twins"] == 50
    assert s["quarantined_a"] == 0
    # every twin is identified; a few may fall back to mask-only (no op-swap)
    assert s["mask_only_twins"] <= 6


def test_all_twins_identified(manifest):
    from progrecon.corpus import identifiability
    for tw in manifest.twins:
        assert identifiability.check(tw, n_probes=300).passed, tw.id


def test_corpus_b_distinct_and_abstract(manifest):
    assert len({t.source for t in manifest.corpus_b}) == 50  # all distinct
    assert all(t.corpus == "B" and not t.semantic for t in manifest.corpus_b)


def test_twins_match_their_a(manifest):
    # Twins are structurally isomorphic to their A (names are masked, so compare
    # the name-invariant structure: node ids, per-node arity, producers by
    # output position, and the complexity tuple).
    for a, tw in zip(manifest.corpus_a, manifest.twins, strict=True):
        assert tw.twin_id == a.id
        assert tw.complexity == a.complexity  # equal complexity tuple
        assert [n.node_id for n in tw.graph.nodes] == [n.node_id for n in a.graph.nodes]
        assert [len(n.inputs) for n in tw.graph.nodes] == [len(n.inputs) for n in a.graph.nodes]
        a_producers = [a.graph.output_map[f.name] for f in a.schema.outputs]
        tw_producers = [tw.graph.output_map[f.name] for f in tw.schema.outputs]
        assert a_producers == tw_producers
        assert tw.semantic is False  # masked / abstract
        assert all(f.name.startswith("in_") for f in tw.schema.inputs)


def test_corpus_a_is_semantic(manifest):
    assert all(t.corpus == "A" and t.semantic and t.domain for t in manifest.corpus_a)
    # at least some fields carry semantic labels
    assert any(f.semantic_label for t in manifest.corpus_a for f in t.schema.inputs)


def test_write_and_reload(manifest):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = CFG.model_copy(update={"paths": CFG.paths.model_copy(update={"manifests_dir": tmp})})
        paths = build_corpus.write_corpus(manifest, cfg=cfg)
        assert Path(paths["summary"]).exists()
        reloaded = build_corpus.load_corpus_jsonl(paths["corpus_a"])
        assert len(reloaded) == 50
        assert reloaded[0] == manifest.corpus_a[0]
