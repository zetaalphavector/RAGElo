from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrievalEvaluatorFormat(BaseModel):
    reasoning: str = Field(..., description="A concise explanation and reasoning of the relevance of the document.")
    score: int = Field(
        ...,
        description="Your relevance score for the document. 0 for non-relevant, 1 for somewhat relevant and 2 for highly relevant.",
    )


class AnswerEvaluatorFormat(BaseModel):
    reasoning: str = Field(..., description="A concise explanation and reasoning of the quality of the answer.")
    score: int = Field(
        ...,
        description="Your scoring for the quality of the answer. 0 if the answer does not answer the question, 1 if the answer answers the question but is not very helpful and 2 if the answer answers the question and is very helpful.",
    )


class PairwiseAnswerEvaluatorFormat(BaseModel):
    answer_a_analysis: str = Field(..., description="A string with your analysis of assistant A's answer")
    answer_b_analysis: str = Field(..., description="A string with your analysis of assistant B's answer")
    comparison_reasoning: str = Field(
        ..., description="A string with your comparison between the two answers and their differences"
    )
    winner: Literal["A", "B", "C"] = Field(..., description="The winner of the pairwise comparison.")


class RDNAMEvaluatorFormat(BaseModel):
    score: float = Field(
        ...,
        description="A number between 0 and 2 representing the relevance score of the document.",
    )
    intent_match: float | None = Field(
        ...,
        description="An number between 0 and 2 representing the match of the document to the query intent.",
    )
    trustworthiness: float | None = Field(
        ..., description="An number between 0 and 2 representing the trustworthiness of the document."
    )


class EloTournamentResult(BaseModel):
    """A class to store the results of an Elo tournament between multiple agents."""

    agents: list[str]
    scores: dict[str, float]
    games_played: dict[str, int]
    wins: dict[str, int]
    loses: dict[str, int]
    ties: dict[str, int]
    std_dev: dict[str, float]
    total_games: int
    total_tournaments: int
