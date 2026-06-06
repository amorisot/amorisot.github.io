"""All pydantic data contracts for the study — the single source of truth.

These interfaces are depended upon by every other module. They are intended to
be *frozen* early (CHECKPOINT 0): changing them later is expensive.

Design notes:
* Every model forbids unknown fields (``extra="forbid"``) so contract typos
  surface immediately rather than silently passing through.
* ``Transform.schema`` is named exactly as the protocol mandates even though it
  shadows the deprecated ``BaseModel.schema`` method; the warning is filtered
  in ``pyproject.toml``.
* ``Transform.source`` is always codegen'd from ``Transform.graph`` (see
  ``dsl.codegen``). The two are never hand-maintained independently.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class _Base(BaseModel):
    """Shared config: forbid unknown fields to catch contract drift."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

FieldType = Literal["int", "float", "categorical", "bool"]


class FieldSpec(_Base):
    """A single input or output field.

    For *inputs*, the distribution spec (``low``/``high`` for numerics,
    ``categories`` for categoricals) describes ``P_x``. For *outputs* the
    distribution spec is ignored — the output type is determined by the
    ground-truth computation.

    Field *names* are masked to ``in_0``/``out_0`` at serialization time for
    abstract tasks (Claim 3 manipulation lives in ``data.serialize``).
    """

    name: str
    type: FieldType
    # Distribution spec for inputs (ignored for outputs):
    low: float | None = None
    high: float | None = None
    categories: list[str] | None = None
    # Present for Corpus A (semantic), None for Corpus B (abstract):
    semantic_label: str | None = None

    @model_validator(mode="after")
    def _check_distribution(self) -> FieldSpec:
        if self.type == "categorical" and self.categories is not None and not self.categories:
            raise ValueError(f"field {self.name!r}: categorical with empty categories list")
        if self.type in ("int", "float") and self.low is not None and self.high is not None:
            if self.low > self.high:
                raise ValueError(f"field {self.name!r}: low {self.low} > high {self.high}")
        return self


class Schema(_Base):
    inputs: list[FieldSpec]  # length n
    outputs: list[FieldSpec]  # length m

    @property
    def n(self) -> int:
        return len(self.inputs)

    @property
    def m(self) -> int:
        return len(self.outputs)

    @model_validator(mode="after")
    def _check_nonempty_unique(self) -> Schema:
        if not self.inputs:
            raise ValueError("schema must declare at least one input field")
        if not self.outputs:
            raise ValueError("schema must declare at least one output field")
        in_names = [f.name for f in self.inputs]
        out_names = [f.name for f in self.outputs]
        if len(set(in_names)) != len(in_names):
            raise ValueError(f"duplicate input field names: {in_names}")
        if len(set(out_names)) != len(out_names):
            raise ValueError(f"duplicate output field names: {out_names}")
        return self


# ---------------------------------------------------------------------------
# Dataflow graph (for twins + complexity + validation + codegen)
# ---------------------------------------------------------------------------


class OpNode(_Base):
    node_id: str
    op: str  # key into ops_spec.OPS_SPEC
    params: dict[str, Any] = Field(default_factory=dict)  # op constants, e.g. {"k": 8}
    inputs: list[str]  # node_ids or input field names, in argument order


class DataflowGraph(_Base):
    nodes: list[OpNode]
    output_map: dict[str, str]  # output field name -> producing node_id

    @model_validator(mode="after")
    def _check_unique_node_ids(self) -> DataflowGraph:
        ids = [nd.node_id for nd in self.nodes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate node_ids: {ids}")
        return self

    def node(self, node_id: str) -> OpNode:
        for nd in self.nodes:
            if nd.node_id == node_id:
                return nd
        raise KeyError(node_id)


# ---------------------------------------------------------------------------
# Complexity
# ---------------------------------------------------------------------------


class ComplexityTuple(_Base):
    n: int  # number of (consumed + distractor) input fields
    m: int  # number of output fields
    depth: int  # longest op-path from any input to any output
    max_arity: int  # max fan-in across nodes
    max_tier: int  # max op tier used
    n_branches: int  # total branch arms across conditional/dispatch nodes


# ---------------------------------------------------------------------------
# Transform (a base transformation)
# ---------------------------------------------------------------------------


class Transform(_Base):
    id: str  # e.g. "B-GEN-007" or "A-PRICE-03"
    corpus: Literal["A", "B"]
    semantic: bool  # True for economic/semantic, False for abstract
    domain: str | None = None  # Corpus A only
    description: str | None = None
    schema: Schema = Field()  # type: ignore[assignment]  # protocol mandates field name `schema` (shadows BaseModel.schema)
    source: str  # ground-truth Python: def transform(x: dict) -> dict  (codegen'd from graph)
    graph: DataflowGraph  # for twin construction + complexity + validation
    complexity: ComplexityTuple
    px_seed: int  # base seed for P_x
    twin_id: str | None = None  # link to matched twin
    provenance: dict[str, Any] = Field(default_factory=dict)  # generator params / authoring metadata


# ---------------------------------------------------------------------------
# Instancing (perturbation)
# ---------------------------------------------------------------------------


class PerturbationSpec(_Base):
    seed: int
    resample_constants: bool = False
    permute_fields: bool = False
    rename_fields: bool = False
    distribution_shift: bool = False
    n_distractors: int = 0  # irrelevant input fields added (clean n-lever for Claim 1)
    branch_coverage_bias: bool = False

    @model_validator(mode="after")
    def _check_distractors(self) -> PerturbationSpec:
        if self.n_distractors < 0:
            raise ValueError("n_distractors must be >= 0")
        return self


class Task(_Base):
    task_id: str  # transform_id + perturbation seed
    transform: Transform  # the (possibly perturbed) concrete transform
    perturbation: PerturbationSpec


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class Example(_Base):
    row_id: int
    x: dict[str, Any]
    y: dict[str, Any]


class SampleSet(_Base):
    kind: Literal["train", "test_id", "test_ood"]
    examples: list[Example]
    seed: int

    @property
    def is_test(self) -> bool:
        return self.kind in ("test_id", "test_ood")

    @property
    def row_ids(self) -> set[int]:
        return {ex.row_id for ex in self.examples}


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class GridCell(_Base):
    n: int
    m: int
    k: int
    corpus: Literal["A", "B", "twin"]
    complexity_band: Literal["low", "med", "high"]
    noise: Literal["none", "eps", "flip"]
    mode: Literal["scratch", "infill"]
    model_id: str
    query_budget: int  # B_q

    def cell_key(self) -> str:
        """Stable identifier for resumability (skip completed cells)."""
        return (
            f"n{self.n}-m{self.m}-k{self.k}-{self.corpus}-{self.complexity_band}"
            f"-{self.noise}-{self.mode}-{self.model_id}-q{self.query_budget}"
        )


class RunOutcome(_Base):
    task_id: str
    cell: GridCell
    repeat_idx: int
    final_source: str | None
    outcome: Literal["solved", "failed", "budget_exhausted", "error"]
    iterations_used: int
    queries_used: int = 0
    tokens_used: int
    usd_cost: float
    transcript_path: str
    # Scoring (filled in later by scoring.metrics):
    exact_pass: bool | None = None
    row_accuracy: float | None = None
    field_accuracy: float | None = None
    ood_exact_pass: bool | None = None
    ood_row_accuracy: float | None = None
