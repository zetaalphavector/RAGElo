from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, computed_field
from pydantic.json_schema import SkipJsonSchema


class EvaluatorResult(BaseModel):
    """Generic class with the results of an evaluator.
    Args:
        qid str: The query ID to which the result corresponds.
        evaluator_name str: The name of the evaluator that produced this result.
        agent str | None: The agent that provided the answer or retrieved the document.
        exception Optional[str]: Any exception captured during evaluation.
    """

    qid: Annotated[str, SkipJsonSchema] = Field(description="The query ID to which the result corresponds.")
    evaluator_name: Annotated[str, SkipJsonSchema] = Field(
        description="The name of the evaluator that produced this result."
    )
    exception: Annotated[str | None, SkipJsonSchema] = Field(
        default=None, description="Any exception captured during evaluation."
    )

    def strigify_answer(self) -> str:
        return self.model_dump_json(indent=4)


class RetrievalEvaluatorResult(EvaluatorResult):
    """Flattened results of a retrieval evaluator.
    Args:
        did str: The document ID to which the result corresponds.
        score Optional[float|int]: Normalized relevance score (e.g., 0-2). If unavailable, may be None.
        reasoning Optional[str]: Reasoning for the score when available.
        intent_match Optional[float]: Optional aspect (RDNAM).
        trustworthiness Optional[float]: Optional aspect (RDNAM).
    """

    did: Annotated[str, SkipJsonSchema] = Field(description="The document ID to which the result corresponds.")
    reasoning: str = Field(..., description="A concise explanation and reasoning of the relevance of the document.")
    score: float | int = Field(
        ...,
        description="Your relevance score for the document. 0 for non-relevant, 1 for somewhat relevant and 2 for highly relevant.",
    )

    def strigify_answer(self) -> str:
        return f"Score: {self.score}\nReasoning: {self.reasoning}"


class RDNAMEvaluatorResult(RetrievalEvaluatorResult):
    """Specialized retrieval result for RDNAM.
    Keeps flattened fields but signals specific evaluator semantics.
    """

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

    def strigify_answer(self) -> str:
        return f"Score: {self.score}\nReasoning: {self.reasoning}\nIntent Match: {self.intent_match}\nTrustworthiness: {self.trustworthiness}"


class AnswerEvaluatorResult(EvaluatorResult):
    """Flattened results of an answer evaluator for a single agent answer.
    Args:
        agent str: The agent that provided the answer.
        score Optional[int]: Quality score.
        reasoning Optional[str]: Reasoning for the score when available.
    """

    agent: Annotated[str, SkipJsonSchema] = Field(description="The agent that provided the answer.")
    reasoning: str = Field(..., description="A concise explanation and reasoning of the quality of the answer.")
    score: int = Field(
        ...,
        description="Your score for the quality of the answer. 0 if the answer does not answer the question, 1 if the answer answers the question but is not very helpful and 2 if the answer answers the question and is very helpful.",
    )

    def strigify_answer(self) -> str:
        return f"Score: {self.score}\nReasoning: {self.reasoning}"


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

    agent_a: Annotated[str, SkipJsonSchema]
    agent_b: Annotated[str, SkipJsonSchema]
    answer_a_analysis: str = Field(..., description="A string with your analysis of assistant A's answer")
    answer_b_analysis: str = Field(..., description="A string with your analysis of assistant B's answer")
    comparison_reasoning: str = Field(
        ..., description="A string with your comparison between the two answers and their differences"
    )
    winner: Literal["A", "B", "C"] = Field(..., description="The winner of the pairwise comparison.")

    def strigify_answer(self) -> str:
        return f"""Answer A Analysis: {self.answer_a_analysis}
Answer B Analysis: {self.answer_b_analysis}
Comparison Reasoning: {self.comparison_reasoning}
Winner: {self.winner}"""

    @computed_field
    @property
    def game_id(self) -> str:
        """The ID of the pairwise game."""
        sorted_agents = sorted([self.agent_a, self.agent_b])
        return f"{sorted_agents[0]}-{sorted_agents[1]}"


class EloTournamentResult(BaseModel):
    """A class to store the results of an Elo tournament between multiple agents."""

    evaluator_name: str = "EloTournament"
    agents: list[str]
    scores: dict[str, float]
    games_played: dict[str, int]
    wins: dict[str, int]
    loses: dict[str, int]
    ties: dict[str, int]
    std_dev: dict[str, float]
    total_games: int
    total_tournaments: int
