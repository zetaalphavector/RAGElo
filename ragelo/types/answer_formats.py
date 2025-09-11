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


class Criterion(BaseModel):
    criterion_name: str = Field(
        description="The name of the criterion to be used to evaluate the quality of the responses."
    )
    supporting_documents: list[str] = Field(
        description="The list of documents IDs that support the criterion. If no documents support the criterion, leave this list empty."
    )
    short_question: str = Field(
        description="A short, yes/no question that can be used to evaluate the quality of the responses."
    )


class CriterionEvaluation(BaseModel):
    criterion: Criterion = Field(..., description="The criterion used for evaluating the answer quality")
    reasoning: str = Field(..., description="The LLM reasoning for the winner of the criteria")
    winner: Literal["A", "B", "C"] = Field(..., description="The winner of the criteria")


class RubricAnswerFormat(BaseModel):
    criteria: list[CriterionEvaluation] = Field(..., description="The criteria used for evaluating the answer quality")
    agent_a_wins: int = Field(..., description="The number of criteria that agent A wins")
    agent_b_wins: int = Field(..., description="The number of criteria that agent B wins")
    equally_good: int = Field(..., description="The number of criteria that agent A and agent B are equally good")
    equally_bad: int = Field(..., description="The number of criteria that agent A and agent B are equally bad")
    winner: Literal["A", "B", "C"] = Field(..., description="The winner of the pairwise comparison")


class CriterionEvaluationPointwise(BaseModel):
    criterion: Criterion = Field(..., description="The criterion used for evaluating the answer quality")
    reasoning: str = Field(..., description="The LLM reasoning for the score of the criterion")
    fulfillment: bool = Field(..., description="Whether the criterion is fully fulfilled by the answer")


class RubricPointwiseAnswerFormat(BaseModel):
    criteria: list[CriterionEvaluationPointwise] = Field(
        ..., description="The criteria used for evaluating the answer quality"
    )
    average_score: float = Field(..., description="The average score of the criteria")
