from __future__ import annotations

from pydantic import BaseModel, ValidationError, model_validator

from ragelo.types.answer_formats import RetrievalEvaluatorFormat


class EvaluatorResult(BaseModel):
    """Generic class with the results of an evaluator.
    Args:
        qid str: The query ID to which the result corresponds.
        agent str | None: The agent that provided the answer or retrieved the document.
        answer Optional[BaseModel]: The processed answer provided by the evaluator.
    """

    qid: str
    evaluator_name: str
    agent: str | None = None
    answer: BaseModel | None = None
    exception: str | None = None

    @model_validator(mode="before")
    @classmethod
    def check_agents(cls, v):
        exception = v.get("exception")
        answer = v.get("answer")
        if answer is None and exception is None:
            raise ValidationError(
                "Either answer or raw_answer must be provided. Otherwise, an exception must be provided."
            )
        return v


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
    def check_agents(cls, v):
        agent = v.get("agent")
        agent_a = v.get("agent_a")
        agent_b = v.get("agent_b")
        if agent is None and agent_a is None and agent_b is None:
            raise ValidationError("Either agent or agent_a and agent_b must be provided")
        if agent_a is not None and agent_b is not None:
            v["pairwise"] = True
        return v


class RetrievalEvaluatorResult(EvaluatorResult):
    """The results of a retrieval evaluator.
    Args:
        did str: The document ID to which the result corresponds.
    """

    did: str
    answer: RetrievalEvaluatorFormat | None = None


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
