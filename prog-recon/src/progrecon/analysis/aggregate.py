"""Logs/manifests -> tidy dataframe (build spec §Module analysis)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..types import RunOutcome


def outcomes_to_frame(outcomes: list[RunOutcome]) -> pd.DataFrame:
    rows = []
    for o in outcomes:
        c = o.cell
        rows.append({
            "task_id": o.task_id,
            "cell_key": c.cell_key(),
            "n": c.n, "m": c.m, "k": c.k, "corpus": c.corpus,
            "complexity_band": c.complexity_band, "noise": c.noise, "mode": c.mode,
            "model_id": c.model_id, "query_budget": c.query_budget,
            "repeat_idx": o.repeat_idx,
            "outcome": o.outcome,
            "exact_pass": o.exact_pass,
            "ood_exact_pass": o.ood_exact_pass,
            "row_accuracy": o.row_accuracy,
            "field_accuracy": o.field_accuracy,
            "ood_row_accuracy": o.ood_row_accuracy,
            "iterations_used": o.iterations_used,
            "queries_used": o.queries_used,
            "tokens_used": o.tokens_used,
            "usd_cost": o.usd_cost,
        })
    return pd.DataFrame(rows)


def load_manifest_dir(path: str | Path) -> list[RunOutcome]:
    p = Path(path)
    outcomes: list[RunOutcome] = []
    for f in sorted(p.glob("*.json")):
        try:
            outcomes.append(RunOutcome.model_validate_json(f.read_text()))
        except (ValueError, json.JSONDecodeError):
            continue  # skip non-RunOutcome json (e.g. transcripts)
    return outcomes


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Group a tidy outcomes frame to cell level (means of the key metrics)."""
    if df.empty:
        return df
    grouped = (
        df.groupby("cell_key")
        .agg(
            n_runs=("task_id", "count"),
            exact_rate=("exact_pass", "mean"),
            ood_exact_rate=("ood_exact_pass", "mean"),
            mean_row_acc=("row_accuracy", "mean"),
            mean_tokens=("tokens_used", "mean"),
            mean_cost=("usd_cost", "mean"),
        )
        .reset_index()
    )
    return grouped
