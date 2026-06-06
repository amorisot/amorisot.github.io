"""Fixed prompt templates (protocol Appendix B).

Deterministic templates only — no per-run randomness in the wording. The agent
communicates with the harness through a small, parseable protocol:

* A fenced ```python block defining ``def transform(x: dict) -> dict`` is a
  candidate program (the harness TESTs it).
* A line ``QUERY: [ {..}, {..} ]`` (JSON list of input dicts) asks the oracle
  for ground-truth outputs on new inputs (only honored if query budget remains).
* ``DONE`` may accompany a final code block; the harness finalizes on a passing
  TEST regardless.
"""

from __future__ import annotations

from ..data import serialize
from ..types import SampleSet, Schema

SYSTEM = """You are reconstructing a hidden deterministic program that maps an \
input record to an output record. You are shown example (input, output) pairs and \
must infer the exact program.

Protocol:
- To propose a program, output exactly one fenced Python code block defining a \
function `transform(x: dict) -> dict`. It must be pure and deterministic and use \
only the Python standard library (`math` is available).
- The keys of the returned dict must be exactly the output field names. The keys \
read from `x` must be exactly the input field names shown.
- If you want ground-truth outputs for specific new inputs, output a single line:
  `QUERY: [{...}, {...}]`  (a JSON list of input records). Only do this if told \
queries are available.
- Propose your best current program every turn. When you are confident, you may \
add a line `DONE`.

Make the program as simple as the data allows; do not overfit to noise."""


def initial_user(
    train_visible: SampleSet,
    schema: Schema,
    *,
    mask_names: bool,
    mode: str,
    infill_source: str | None,
    query_budget: int,
) -> str:
    parts = [
        "Reconstruct the program behind these examples.",
        "",
        serialize.for_prompt(train_visible, schema, mask_names=mask_names),
        "",
    ]
    if query_budget > 0:
        parts.append(
            f"You may issue up to {query_budget} QUERY requests for new inputs.\n"
        )
    else:
        parts.append("No queries are available; infer from the examples alone.\n")
    if mode == "infill" and infill_source:
        parts.append(
            "You are given a partial implementation to COMPLETE (infill mode). "
            "Fill in the missing logic and return the full function:\n"
            f"```python\n{infill_source}\n```\n"
        )
    parts.append("Output your `transform` function now.")
    return "\n".join(parts)


def test_feedback(n_fail: int, n_total: int, failures: list[str]) -> str:
    head = (
        f"Your program failed {n_fail}/{n_total} held-out training rows. "
        "Examples of mismatches (input -> your output vs expected):"
    )
    return head + "\n" + "\n".join(failures) + "\nRevise and output the corrected `transform`."


def query_results(results_text: str) -> str:
    return "Ground-truth outputs for your queried inputs:\n" + results_text + "\nContinue."


def no_code_feedback() -> str:
    return (
        "I did not find a Python code block. Output exactly one fenced ```python block "
        "defining `transform(x: dict) -> dict`."
    )
