from __future__ import annotations

from typing import Literal

from pydantic import Field

from ragelo.types.answer_formats import Criterion
from ragelo.types.configurations.base_configs import GuidelinesConfigMixin

RubricSource = Literal["documents", "reference_answer"]


class RubricGeneratorConfig(GuidelinesConfigMixin):
    expert_in: str = Field(description="What the LLM should mimic being an expert in.")
    company: str | None = Field(
        default=None,
        description="Name of the company or organization that the user that "
        "submitted the query works for. that the domain belongs to. "
        "(e.g.: ChemCorp, CS Inc.)",
    )
    n_criteria: int = Field(default=5, description="The number of criteria to write.")
    source: RubricSource = Field(
        default="documents",
        description="What the criteria are derived from. 'documents' uses the query's retrieved documents, "
        "so the rubric is bounded by what retrieval found. 'reference_answer' uses query.reference_answer, "
        "which keeps the criteria out of reach of the systems being scored.",
    )
    documents_limit: int = Field(
        default=20,
        description="How many retrieved documents the 'documents' source shows the LLM, best-scored first. "
        "A pooled experiment can hold far more text per query than a rubric prompt should carry.",
    )


class RubricConfigMixin(GuidelinesConfigMixin):
    """The rubric-generation settings shared by every evaluator that grades against `query.rubric`."""

    expert_in: str = Field(description="What the LLM should mimic being an expert in.")
    company: str | None = Field(
        default=None,
        description="Name of the company or organization that the user that "
        "submitted the query works for. that the domain belongs to. "
        "(e.g.: ChemCorp, CS Inc.)",
    )
    n_criteria: int = Field(default=5, description="The number of criteria to use for the evaluator.")
    rubrics: dict[str, list[Criterion]] | None = Field(
        default=None,
        description=(
            "The cache of criteria for the evaluator. Maps a query_id to a list of Criterion objects. "
            "If provided, the evaluator will skip creating the rubric based on the retrieved documents "
            "and use this instead."
        ),
    )
    rubric_source: RubricSource = Field(
        default="documents",
        description="What generated rubrics are derived from, for queries that have none.",
    )
    rubric_documents_limit: int = Field(
        default=20,
        description="How many retrieved documents rubric generation from 'documents' shows the LLM, "
        "best-scored first.",
    )
