from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema


class EvaluationAnswer(BaseModel):
    """Base class for LLM-generated evaluation content (without metadata)."""

    pass


class RetrievalEvaluationAnswer(EvaluationAnswer):
    """Output format for evaluating the relevance of a document to a question."""

    reasoning: str = Field(..., description="A concise explanation and reasoning of the relevance of the document.")
    score: float | int = Field(
        ...,
        description="Your relevance score for the document. 0 for non-relevant, 1 for somewhat relevant "
        "and 2 for highly relevant.",
    )


class AnswerEvaluationAnswer(EvaluationAnswer):
    """Output format for evaluating the quality of an answer to a question."""

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
    """Output format for evaluating the quality of answers from two agents to the same question."""

    answer_a_analysis: str = Field(..., description="A string with your analysis of the answer from agent A")
    answer_b_analysis: str = Field(..., description="A string with your analysis of the answer from agent B")
    comparison_reasoning: str = Field(
        ..., description="A string with your comparison between the two answers and their differences"
    )
    winner: Literal["A", "B", "C"] = Field(
        ...,
        description=(
            "The winner of the pairwise comparison. 'A' if the answer from agent A is better, "
            "'B' if the answer from agent B is better, or 'C' for a tie."
        ),
    )


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
    """Output format for evaluating the relevance of a document to a question."""

    reasoning: Annotated[str, SkipJsonSchema] = ""
    score: float = Field(
        ...,
        description="An number between 0 and 2 representing the overall score of the document.",
    )


class RDNAMMultipleAnnotatorsAnswer(RetrievalEvaluationAnswer):
    """Output format for evaluating the relevance of a document to a question by simulating 5 annotators."""

    score: Annotated[float | int, SkipJsonSchema] = 0.0
    reasoning: Annotated[str, SkipJsonSchema] = ""
    annotator_1: RDNAMEvaluationAnswer
    annotator_2: RDNAMEvaluationAnswer
    annotator_3: RDNAMEvaluationAnswer
    annotator_4: RDNAMEvaluationAnswer
    annotator_5: RDNAMEvaluationAnswer


class RDNAMMultipleAnnotatorsNoAspectsAnswer(RetrievalEvaluationAnswer):
    """Output format for evaluating the relevance of a document to a question by simulating 5 annotators."""

    score: Annotated[float | int, SkipJsonSchema] = 0.0
    reasoning: Annotated[str, SkipJsonSchema] = ""
    annotator_1: RDNAMNoAspectsAnswer
    annotator_2: RDNAMNoAspectsAnswer
    annotator_3: RDNAMNoAspectsAnswer
    annotator_4: RDNAMNoAspectsAnswer
    annotator_5: RDNAMNoAspectsAnswer


class Criterion(BaseModel):
    criterion_name: str = Field(
        description="The name of the criterion to be used to evaluate the quality of the responses."
    )
    evidence: list[str] = Field(
        description="The list of documents IDs or snippets that support the criterion. "
        "If no documents support the criterion, leave this list empty."
    )
    short_question: str = Field(
        description="A short, yes/no question that can be used to evaluate the quality of the responses."
    )


class CriterionEvaluation(BaseModel):
    criterion: Criterion = Field(..., description="The criterion used for evaluating the answer quality")
    reasoning: str = Field(..., description="The LLM reasoning for the winner of the criteria")
    winner: Literal["A", "B", "C"] = Field(..., description="The winner of the criteria")


class RubricAnswerFormat(EvaluationAnswer):
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


class RubricPointwiseAnswerFormat(EvaluationAnswer):
    criteria: list[CriterionEvaluationPointwise] = Field(
        ..., description="The criteria used for evaluating the answer quality"
    )
    average_score: float = Field(..., description="The average score of the criteria")


class RubricSchema(BaseModel):
    criteria: list[Criterion] = Field(description="The criteria to be used to evaluate the quality of the responses.")
