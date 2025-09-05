from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PairWiseAnswerAnswerFormat(BaseModel):
    answer_a_analysis: str = Field(..., description="A string with your analysis of assistant A's answer")
    answer_b_analysis: str = Field(..., description="A string with your analysis of assistant B's answer")
    comparison_reasoning: str = Field(
        ..., description="A string with your comparison between the two answers and their differences"
    )
    winner: Literal["A", "B", "C"] = Field(..., description="The winner of the pairwise comparison.")


class RetrievalAnswerEvaluatorFormat(BaseModel):
    reasoning: str = Field(..., description="A concise explanation and reasoning of the relevance of the document.")
    score: int = Field(
        ...,
        description="Your relevance score for the document. 0 for non-relevant, 1 for somewhat relevant and 2 for highly relevant.",
    )


class RDNAMAnswerEvaluatorFormat(BaseModel):
    overall: float = Field(
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


class RDNAMAnswerNoAspects(BaseModel):
    overall: float = Field(
        ...,
        description="An number between 0 and 2 representing the score of the document.",
    )


class RDNAMMultipleAnnotatorsAnswer(BaseModel):
    annotator_1: RDNAMAnswerEvaluatorFormat
    annotator_2: RDNAMAnswerEvaluatorFormat
    annotator_3: RDNAMAnswerEvaluatorFormat
    annotator_4: RDNAMAnswerEvaluatorFormat
    annotator_5: RDNAMAnswerEvaluatorFormat


class RDNAMMultipleAnnotatorsAnswerNoAspects(BaseModel):
    annotator_1: RDNAMAnswerNoAspects
    annotator_2: RDNAMAnswerNoAspects
    annotator_3: RDNAMAnswerNoAspects
    annotator_4: RDNAMAnswerNoAspects
    annotator_5: RDNAMAnswerNoAspects
