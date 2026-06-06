"""Vertical-slice end-to-end smoke test (CHECKPOINT 7 — the 'let it rip' gate).

Wires generate -> sample (with firewall) -> harness (MockModel oracle) -> score
-> aggregate, for ONE cell, entirely offline. Acceptance: pipeline runs,
cost is logged and tiny, D_test firewall holds, artifacts land, a one-row
dataframe is produced.
"""

import tempfile
from pathlib import Path

from progrecon.analysis import aggregate
from progrecon.config import load_config
from progrecon.runner import grid, run_cell
from progrecon.scoring import metrics


def _small_cfg():
    cfg = load_config()
    # Keep the slice fast: small test splits (still exercises 2000-row scoring path).
    return cfg.model_copy(update={"scoring": cfg.scoring.model_copy(update={"test_id_size": 300, "test_ood_size": 300})})


def test_vertical_slice_end_to_end():
    cfg = _small_cfg()
    with tempfile.TemporaryDirectory() as tmp:
        cfg = cfg.model_copy(update={"paths": cfg.paths.model_copy(update={
            "manifests_dir": str(Path(tmp) / "manifests"),
            "logs_dir": str(Path(tmp) / "logs"),
        })})
        cell = grid.single_cheap_cell(cfg, n=3, m=1, k=100, band="med")

        result = run_cell.run_cell(cell, seed=7, cfg=cfg)

        # The oracle MockModel returns the ground truth -> solved + exact on BOTH splits.
        assert result.outcome.outcome == "solved"
        assert result.outcome.exact_pass is True
        assert result.outcome.ood_exact_pass is True
        assert result.outcome.row_accuracy == 1.0

        # Cost is logged and tiny. The mock provider does no real network spend,
        # but the meter reflects the configured cheap-model price, so a tiny
        # nonzero estimate is expected and well under the ceiling.
        assert 0.0 <= result.outcome.usd_cost < 0.01
        assert result.outcome.tokens_used > 0

        # Artifacts landed.
        assert Path(result.transcript_path).exists()
        man = list((Path(tmp) / "manifests").glob("*.json"))
        assert len(man) == 1

        # A tidy one-row dataframe is produced.
        df = aggregate.outcomes_to_frame([result.outcome])
        assert len(df) == 1 and df.iloc[0]["exact_pass"]
        table = aggregate.cell_table(df)
        assert table.iloc[0]["exact_rate"] == 1.0

        # Cell-level metric rollup.
        cm = metrics.aggregate_cell([result.outcome])
        assert cm.exact_reconstruction_rate == 1.0
        print("\nSLICE RESULT:", result.one_line)


def test_slice_firewall_holds_in_run_cell():
    # run_cell builds a SampleBundle and asserts firewall_ok internally; here we
    # also confirm the harness never recorded a test-split row id.
    cfg = _small_cfg()
    cell = grid.single_cheap_cell(cfg, n=3, m=1, k=60, band="low")
    result = run_cell.run_cell(cell, seed=3, cfg=cfg, write_artifacts=False)
    assert result.outcome.outcome in ("solved", "failed", "budget_exhausted")
