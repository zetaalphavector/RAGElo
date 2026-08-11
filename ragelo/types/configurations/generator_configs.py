from __future__ import annotations

from typing import Literal

from pydantic import Field

from ragelo.types.configurations.base_configs import BaseConfig

RubricSource = Literal["documents", "reference_answer"]


class RubricGeneratorConfig(BaseConfig):
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
