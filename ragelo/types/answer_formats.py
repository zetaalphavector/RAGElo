from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

PairwiseWinner = Literal["A", "B", "C"]


def swap_pairwise_winner(winner: PairwiseWinner) -> PairwiseWinner:
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return "C"


def swap_pairwise_labels(text: str) -> str:
    swaps = {
        "[[A]]": "[[B]]",
        "[[B]]": "[[A]]",
        "agent A": "agent B",
        "agent B": "agent A",
        "Agent A": "Agent B",
        "Agent B": "Agent A",
        "assistant A": "assistant B",
        "assistant B": "assistant A",
        "Assistant A": "Assistant B",
        "Assistant B": "Assistant A",
        "answer A": "answer B",
        "answer B": "answer A",
        "Answer A": "Answer B",
        "Answer B": "Answer A",
        "response A": "response B",
        "response B": "response A",
        "Response A": "Response B",
        "Response B": "Response A",
        "A": "B",
        "B": "A",
    }
    pattern = re.compile(
        r"\[\[A\]\]|\[\[B\]\]|\bagent A\b|\bagent B\b|\bAgent A\b|\bAgent B\b|"
        r"\bassistant A\b|\bassistant B\b|\bAssistant A\b|\bAssistant B\b|\banswer A\b|"
        r"\banswer B\b|\bAnswer A\b|\bAnswer B\b|\bresponse A\b|\bresponse B\b|"
        r"\bResponse A\b|\bResponse B\b|\bA\b|\bB\b"
    )
    return pattern.sub(lambda match: swaps[match.group(0)], text)


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

    answer_a_strengths: list[str] = Field(
        default_factory=list,
        description="Short strengths of [[A]]. Use only [[A]] and [[B]] when referring to the assistants.",
    )
    answer_a_weaknesses: list[str] = Field(
        default_factory=list,
        description="Short weaknesses of [[A]]. Use only [[A]] and [[B]] when referring to the assistants.",
    )
    answer_b_strengths: list[str] = Field(
        default_factory=list,
        description="Short strengths of [[B]]. Use only [[A]] and [[B]] when referring to the assistants.",
    )
    answer_b_weaknesses: list[str] = Field(
        default_factory=list,
        description="Short weaknesses of [[B]]. Use only [[A]] and [[B]] when referring to the assistants.",
    )
    answer_a_analysis: str = Field(..., description="A string with your analysis of the answer from agent A")
    answer_b_analysis: str = Field(..., description="A string with your analysis of the answer from agent B")
    comparison_reasoning: str = Field(
        ..., description="A string with your comparison between the two answers and their differences"
    )
    winner_reasoning: str = Field(
        default="",
        description=(
            "A concise explanation of why the winner was chosen. Use only [[A]] and [[B]] when referring to the "
            "assistants."
        ),
    )
    winner: PairwiseWinner = Field(
        ...,
        description=(
            "The winner of the pairwise comparison. 'A' if the answer from agent A is better, "
            "'B' if the answer from agent B is better, or 'C' for a tie."
        ),
    )

    def swap_perspective(self) -> Self:
        return self.model_copy(
            update={
                "answer_a_strengths": [swap_pairwise_labels(item) for item in self.answer_b_strengths],
                "answer_a_weaknesses": [swap_pairwise_labels(item) for item in self.answer_b_weaknesses],
                "answer_b_strengths": [swap_pairwise_labels(item) for item in self.answer_a_strengths],
                "answer_b_weaknesses": [swap_pairwise_labels(item) for item in self.answer_a_weaknesses],
                "answer_a_analysis": swap_pairwise_labels(self.answer_b_analysis),
                "answer_b_analysis": swap_pairwise_labels(self.answer_a_analysis),
                "comparison_reasoning": swap_pairwise_labels(self.comparison_reasoning),
                "winner_reasoning": swap_pairwise_labels(self.winner_reasoning),
                "winner": swap_pairwise_winner(self.winner),
            }
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
    weight: float | None = Field(
        default=None,
        description="The relative importance of this criterion. "
        "Higher values give more weight in the final score. If not provided, all criteria are weighted equally.",
    )


class CriterionEvaluation(BaseModel):
    criterion: Criterion = Field(..., description="The criterion used for evaluating the answer quality")
    agent_a_assessment: str = Field(
        default="",
        description="How well [[A]] satisfies the criterion. Use only [[A]] and [[B]] in text.",
    )
    agent_b_assessment: str = Field(
        default="",
        description="How well [[B]] satisfies the criterion. Use only [[A]] and [[B]] in text.",
    )
    winner_reasoning: str = Field(
        default="",
        description="Why the criterion winner was chosen. Use only [[A]] and [[B]] in text.",
    )
    reasoning: str = Field(..., description="The LLM reasoning for the winner of the criteria")
    winner: PairwiseWinner = Field(..., description="The winner of the criteria")

    def swap_perspective(self) -> Self:
        return self.model_copy(
            update={
                "agent_a_assessment": swap_pairwise_labels(self.agent_b_assessment),
                "agent_b_assessment": swap_pairwise_labels(self.agent_a_assessment),
                "winner_reasoning": swap_pairwise_labels(self.winner_reasoning),
                "reasoning": swap_pairwise_labels(self.reasoning),
                "winner": swap_pairwise_winner(self.winner),
            }
        )


class RubricAnswerFormat(EvaluationAnswer):
    criteria: list[CriterionEvaluation] = Field(..., description="The criteria used for evaluating the answer quality")
    agent_a_wins: float = Field(..., description="The weighted score of criteria that agent A wins")
    agent_b_wins: float = Field(..., description="The weighted score of criteria that agent B wins")
    equally_good: float = Field(
        ..., description="The weighted score of criteria that agent A and agent B are equally good"
    )
    equally_bad: float = Field(
        ..., description="The weighted score of criteria that agent A and agent B are equally bad"
    )
    winner: PairwiseWinner = Field(..., description="The winner of the pairwise comparison")

    def swap_perspective(self) -> Self:
        return self.model_copy(
            update={
                "criteria": [criterion.swap_perspective() for criterion in self.criteria],
                "agent_a_wins": self.agent_b_wins,
                "agent_b_wins": self.agent_a_wins,
                "winner": swap_pairwise_winner(self.winner),
            }
        )


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
