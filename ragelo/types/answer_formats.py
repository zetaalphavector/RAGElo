from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


class EvaluationAnswer(BaseModel):
    """Base class for LLM-generated evaluation content (without metadata)."""

    pass


class RetrievalEvaluationAnswer(EvaluationAnswer):
    """LLM-generated evaluation for retrieval tasks."""

    reasoning: str = Field(..., description="A concise explanation and reasoning of the relevance of the document.")
    score: float | int = Field(
        ...,
        description="Your relevance score for the document. 0 for non-relevant, 1 for somewhat relevant "
        "and 2 for highly relevant.",
    )


class AnswerEvaluationAnswer(EvaluationAnswer):
    """LLM-generated evaluation for answer quality tasks."""

    reasoning: str = Field(..., description="A concise explanation and reasoning of the quality of the answer.")
    score: int = Field(
        ...,
        description=(
            "Your score for the quality of the answer. 0 if the answer does not answer the question, "
            "1 if the answer answers the question but is not very helpful and 2 if the answer answers the question "
            "and is very helpful."
        ),
    )


class PairwiseEvaluationAnswer(EvaluationAnswer):
    """LLM-generated evaluation for pairwise comparison tasks."""

    answer_a_analysis: str = Field(..., description="A string with your analysis of assistant A's answer")
    answer_b_analysis: str = Field(..., description="A string with your analysis of assistant B's answer")
    comparison_reasoning: str = Field(
        ..., description="A string with your comparison between the two answers and their differences"
    )
    winner: Literal["A", "B", "C"] = Field(..., description="The winner of the pairwise comparison.")


class RDNAMEvaluationAnswer(RetrievalEvaluationAnswer):
    """LLM-generated evaluation for RDNAM retrieval tasks."""

    reasoning: Annotated[str, SkipJsonSchema] = ""
    score: float = Field(
        ...,
        description="An number between 0 and 2 representing the score of the document.",
    )
    intent_match: float | None = Field(
        ...,
        description="An number between 0 and 2 representing the match of the document to the query intent.",
    )
    trustworthiness: float | None = Field(
        ..., description="An number between 0 and 2 representing the trustworthiness of the document."
    )


class RDNAMNoAspectsAnswer(RetrievalEvaluationAnswer):
    """LLM-generated evaluation for RDNAM without aspects."""

    reasoning: Annotated[str, SkipJsonSchema] = ""
    score: float = Field(
        ...,
        description="An number between 0 and 2 representing the overall score of the document.",
    )


class RDNAMMultipleAnnotatorsAnswer(RetrievalEvaluationAnswer):
    """LLM-generated evaluation simulating 5 RDNAM annotators."""

    score: Annotated[float | int, SkipJsonSchema] = 0.0
    reasoning: Annotated[str, SkipJsonSchema] = ""
    annotator_1: RDNAMEvaluationAnswer
    annotator_2: RDNAMEvaluationAnswer
    annotator_3: RDNAMEvaluationAnswer
    annotator_4: RDNAMEvaluationAnswer
    annotator_5: RDNAMEvaluationAnswer


class RDNAMMultipleAnnotatorsNoAspectsAnswer(RetrievalEvaluationAnswer):
    """LLM-generated evaluation simulating 5 RDNAM annotators without aspects."""

    score: Annotated[float | int, SkipJsonSchema] = 0.0
    reasoning: Annotated[str, SkipJsonSchema] = ""
    annotator_1: RDNAMNoAspectsAnswer
    annotator_2: RDNAMNoAspectsAnswer
    annotator_3: RDNAMNoAspectsAnswer
    annotator_4: RDNAMNoAspectsAnswer
    annotator_5: RDNAMNoAspectsAnswer
