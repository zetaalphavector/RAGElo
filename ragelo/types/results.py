from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError, model_validator


class EvaluatorResult(BaseModel):
    """Generic class with the results of an evaluator.
    Args:
        qid str: The query ID to which the result corresponds.
        agent str | None: The agent that provided the answer or retrieved the document.
        raw_answer str | None: The raw answer provided by the LLMProvider used in the evaluator.
        answer Optional[Union[int, str, dict[str, Any]]]: The processed answer provided by the evaluator.
            If the evaluator uses "multi_field_json" as answer_format, this will be a dictionary with evaluation keys.
    """

    qid: str
    agent: str | None = None
    raw_answer: str | None = None
    answer: float | str | dict[str, Any] | BaseModel | None = None
    exception: str | None = None

    @model_validator(mode="before")
    @classmethod
    def check_agents(cls, data: dict[str, Any]):
        exception = data.get("exception")
        raw_answer = data.get("raw_answer")
        answer = data.get("answer")
        if (raw_answer is None or answer is None) and exception is None:
            raise ValidationError(
                "Either answer or raw_answer must be provided. Otherwise, an exception must be provided."
            )
        return data


class AnswerEvaluatorResult(EvaluatorResult):
    """The results of an answer evaluator.
    Args:
        agent str | None: The agent that provided the answer. Only used if pairwise=False.
        agent_a str | None: The first agent that provided the answer. Only used if the evaluator is pairwise.
        agent_b str | None: The second agent that provided the answer. Only used if the evaluator is pairwise.
        pairwise bool: Whether the evaluation is pairwise or not.
    """

    agent: str | None = None
    agent_a: str | None = None
    agent_b: str | None = None
    pairwise: bool = False

    @model_validator(mode="before")
    @classmethod
    def check_agents(cls, data: dict[str, Any]):
        agent = data.get("agent")
        agent_a = data.get("agent_a")
        agent_b = data.get("agent_b")
        if agent is None and agent_a is None and agent_b is None:
            raise ValidationError("Either agent or agent_a and agent_b must be provided")
        if agent_a is not None and agent_b is not None:
            data["pairwise"] = True
        return data


class RetrievalEvaluatorResult(EvaluatorResult):
    """The results of a retrieval evaluator.
    Args:
        did str: The document ID to which the result corresponds.
    """

    did: str


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
