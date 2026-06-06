"""Fixed serialization of a sample set for the prompt.

ONE deterministic serialization (a pretty table + a raw JSON block). The
``mask_names`` lever (Claim 3) strips semantic-label annotations; the x/y keys
are always the literal schema field names so a candidate keyed by them scores
correctly. The abstract condition pairs ``mask_names=True`` with an abstract
(twin) schema whose names are already generic ``in_i``/``out_i``.

HARD FIREWALL: only the ``train`` split may ever be serialized into a prompt.
``for_prompt`` refuses anything else.
"""

from __future__ import annotations

import json

from ..types import SampleSet, Schema


def _field_line(name: str, ftype: str, label: str | None, mask: bool) -> str:
    base = f"- {name}: {ftype}"
    if not mask and label:
        base += f"  ({label})"
    return base


def field_descriptions(schema: Schema, *, mask_names: bool) -> str:
    lines = ["Input fields:"]
    for f in schema.inputs:
        extra = ""
        if f.type == "categorical" and f.categories:
            extra = f" in {sorted(f.categories)}"
        lines.append(_field_line(f.name, f.type, f.semantic_label, mask_names) + extra)
    lines.append("Output fields:")
    for f in schema.outputs:
        lines.append(_field_line(f.name, f.type, f.semantic_label, mask_names))
    return "\n".join(lines)


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return repr(round(v, 6))
    return str(v)


def _table(sampleset: SampleSet, schema: Schema) -> str:
    in_names = [f.name for f in schema.inputs]
    out_names = [f.name for f in schema.outputs]
    header = "| " + " | ".join(in_names) + " | → | " + " | ".join(out_names) + " |"
    sep = "|" + "|".join(["---"] * (len(in_names) + 1 + len(out_names))) + "|"
    rows = [header, sep]
    for ex in sampleset.examples:
        cells = [_fmt(ex.x.get(n)) for n in in_names] + ["→"] + [_fmt(ex.y.get(n)) for n in out_names]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def for_prompt(sampleset: SampleSet, schema: Schema, *, mask_names: bool) -> str:
    """Render the train sample set as a deterministic prompt payload."""
    if sampleset.kind != "train":
        raise ValueError(
            f"refusing to serialize a non-train split into a prompt (kind={sampleset.kind!r}); "
            "test data must never reach the model"
        )
    examples_json = json.dumps(
        [{"x": ex.x, "y": ex.y} for ex in sampleset.examples],
        sort_keys=True,
        default=lambda o: round(o, 6) if isinstance(o, float) else o,
    )
    parts = [
        field_descriptions(schema, mask_names=mask_names),
        "",
        f"Examples ({len(sampleset.examples)} rows):",
        _table(sampleset, schema),
        "",
        "Examples (JSON):",
        examples_json,
    ]
    return "\n".join(parts)
