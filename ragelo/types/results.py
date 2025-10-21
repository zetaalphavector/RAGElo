from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ragelo.types.answer_formats import AnswerEvaluatorFormat, PairwiseAnswerEvaluatorFormat, RetrievalEvaluatorFormat


class EvaluatorResult(BaseModel):
    """Generic class with the results of an evaluator.
    Args:
        qid str: The query ID to which the result corresponds.
        evaluator_name str: The name of the evaluator that produced this result.
        agent str | None: The agent that provided the answer or retrieved the document.
        exception Optional[str]: Any exception captured during evaluation.
    """

    qid: str = Field(
        ...,
        description="The query ID to which the result corresponds.",
    )
    evaluator_name: str = Field(
        default="unknown",
        description="The name of the evaluator that produced this result.",
    )
    agent: str | None = None
    exception: str | None = None


class RetrievalEvaluatorResult(EvaluatorResult):
    """Flattened results of a retrieval evaluator.
    Args:
        did str: The document ID to which the result corresponds.
        score Optional[float|int]: Normalized relevance score (e.g., 0-2). If unavailable, may be None.
        reasoning Optional[str]: Reasoning for the score when available.
        intent_match Optional[float]: Optional aspect (RDNAM).
        trustworthiness Optional[float]: Optional aspect (RDNAM).
    """

    did: str
    score: float | int | None = None
    reasoning: str | None = None


class RDNAMEvaluatorResult(RetrievalEvaluatorResult):
    """Specialized retrieval result for RDNAM.
    Keeps flattened fields but signals specific evaluator semantics.
    """

    intent_match: float | None = None
    trustworthiness: float | None = None


class AnswerEvaluatorResult(EvaluatorResult):
    """Flattened results of an answer evaluator for a single agent answer.
    Args:
        agent str: The agent that provided the answer.
        score Optional[int]: Quality score.
        reasoning Optional[str]: Reasoning for the score when available.
    """

    agent: str
    score: int | None = None
    reasoning: str | None = None


class PairwiseGameEvaluatorResult(EvaluatorResult):
    """Flattened results of a pairwise game evaluator.
    Args:
        agent_a str: The first agent that provided the answer.
        agent_b str: The second agent that provided the answer.
        answer_a_analysis Optional[str]
        answer_b_analysis Optional[str]
        comparison_reasoning Optional[str]
        winner Literal["A","B","C"]
    """

    agent_a: str
    agent_b: str
    answer_a_analysis: str | None = None
    answer_b_analysis: str | None = None
    comparison_reasoning: str | None = None
    winner: Literal["A", "B", "C"] | None = None


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
