from __future__ import annotations

from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, Field, ValidationError, computed_field, field_serializer, model_validator
from pydantic.json_schema import SkipJsonSchema

from ragelo.types.answer_formats import (
    AnswerEvaluationAnswer,
    EvaluationAnswer,
    PairwiseEvaluationAnswer,
    RDNAMEvaluationAnswer,
    RDNAMMultipleAnnotatorsAnswer,
    RDNAMMultipleAnnotatorsNoAspectsAnswer,
    RDNAMNoAspectsAnswer,
    RetrievalEvaluationAnswer,
    RubricAnswerFormat,
    RubricCoverageAnswerFormat,
    RubricPointwiseAnswerFormat,
)


def _resolve_legacy_answer(data: Any, members: tuple[type[EvaluationAnswer], ...]) -> Any:
    """Tag a serialized answer written before `answer_format` existed.

    Tagged payloads are handed to the discriminated union untouched. Untagged ones are matched
    structurally against `members`, which must therefore be ordered most-constrained first: a
    format that declares a superset of another's fields would otherwise validate as the looser one
    and lose the surplus. Formats defined outside the library carry no tag and simply do not match,
    leaving the payload for the field annotation to coerce.
    """
    if not isinstance(data, dict) or data.get("answer") is None:
        return data
    answer_data = data["answer"]
    if not isinstance(answer_data, dict) or isinstance(answer_data, BaseModel):
        return data
    if answer_data.get("answer_format"):
        return data
    for member in members:
        try:
            data["answer"] = member.model_validate(answer_data)
        except ValidationError:
            continue
        break
    return data


class EvaluatorResult(BaseModel):
    """Generic class with the results of an evaluator.
    Args:
        qid str: The query ID to which the result corresponds.
        evaluator_name str: The name of the evaluator that produced this result.
        exception Optional[str]: Any exception captured during evaluation.
        answer: The LLM-generated evaluation content (without metadata).
    """

    qid: SkipJsonSchema[str] = Field(description="The query ID to which the result corresponds.")
    evaluator_name: SkipJsonSchema[str] = Field(description="The name of the evaluator that produced this result.")
    exception: SkipJsonSchema[str | None] = Field(
        default=None, description="Any exception captured during evaluation."
    )
    answer: EvaluationAnswer | None = Field(default=None, description="The LLM-generated evaluation content.")

    @field_serializer("answer")
    def serialize_answer(self, answer: EvaluationAnswer | None, _info) -> dict[str, Any] | None:
        """Serialize the answer field using its actual runtime type, not the base type."""
        if answer is None:
            return None
        # Use the actual class's model_dump to get all fields
        return answer.model_dump()

    def strigify_answer(self) -> str:
        return self.model_dump_json(indent=4)


class RetrievalEvaluatorResult(EvaluatorResult):
    """Results of a retrieval evaluator.
    Args:
        did str: The document ID to which the result corresponds.
        answer: The LLM-generated evaluation with score and reasoning (typically RetrievalEvaluationAnswer).
    """

    did: SkipJsonSchema[str] = Field(description="The document ID to which the result corresponds.")
    answer: (
        Annotated[RetrievalEvaluationAnswer | RubricCoverageAnswerFormat, Field(discriminator="answer_format")] | None
    ) = None

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        return _resolve_legacy_answer(data, (RubricCoverageAnswerFormat, RetrievalEvaluationAnswer))

    @property
    def score(self) -> float | int | None:
        """Convenience property to access the score from the nested answer."""
        if self.answer and hasattr(self.answer, "score"):
            return self.answer.score  # type: ignore
        return None

    @property
    def reasoning(self) -> str | None:
        """Convenience property to access the reasoning from the nested answer."""
        if self.answer and hasattr(self.answer, "reasoning"):
            return self.answer.reasoning  # type: ignore
        return None

    def strigify_answer(self) -> str:
        if self.answer and hasattr(self.answer, "score") and hasattr(self.answer, "reasoning"):
            return f"Score: {self.answer.score}\nReasoning: {self.answer.reasoning}"  # type: ignore
        return "No answer available"


class AnswerEvaluatorResult(EvaluatorResult):
    """Results of an answer evaluator for a single agent answer.
    Args:
        agent str: The agent that provided the answer.
        answer: The LLM-generated evaluation with score and reasoning (typically AnswerEvaluationAnswer).
    """

    agent: SkipJsonSchema[str] = Field(description="The agent that provided the answer.")
    answer: (
        Annotated[AnswerEvaluationAnswer | RubricPointwiseAnswerFormat, Field(discriminator="answer_format")] | None
    ) = None

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        return _resolve_legacy_answer(data, (RubricPointwiseAnswerFormat, AnswerEvaluationAnswer))

    @property
    def score(self) -> int | None:
        """Convenience property to access the score from the nested answer."""
        if self.answer and hasattr(self.answer, "score"):
            return self.answer.score  # type: ignore
        return None

    @property
    def reasoning(self) -> str | None:
        """Convenience property to access the reasoning from the nested answer."""
        if self.answer and hasattr(self.answer, "reasoning"):
            return self.answer.reasoning  # type: ignore
        return None

    def strigify_answer(self) -> str:
        if self.answer and hasattr(self.answer, "score") and hasattr(self.answer, "reasoning"):
            return f"Score: {self.answer.score}\nReasoning: {self.answer.reasoning}"  # type: ignore
        return "No answer available"


class PairwiseGameEvaluatorResult(EvaluatorResult):
    """Results of a pairwise game evaluator.
    Args:
        agent_a str: The first agent that provided the answer.
        agent_b str: The second agent that provided the answer.
        answer: The LLM-generated pairwise evaluation (typically PairwiseEvaluationAnswer).
    """

    agent_a: SkipJsonSchema[str]
    agent_b: SkipJsonSchema[str]
    answer: Annotated[PairwiseEvaluationAnswer | RubricAnswerFormat, Field(discriminator="answer_format")] | None = (
        None
    )
    a_vs_b_result: PairwiseGameEvaluatorResult | None = None
    b_vs_a_result: PairwiseGameEvaluatorResult | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        return _resolve_legacy_answer(data, (RubricAnswerFormat, PairwiseEvaluationAnswer))

    @property
    def answer_a_analysis(self) -> str | None:
        """Convenience property to access answer_a_analysis from the nested answer."""
        if isinstance(self.answer, PairwiseEvaluationAnswer):
            return self.answer.answer_a_analysis
        return None

    @property
    def answer_b_analysis(self) -> str | None:
        """Convenience property to access answer_b_analysis from the nested answer."""
        if isinstance(self.answer, PairwiseEvaluationAnswer):
            return self.answer.answer_b_analysis
        return None

    @property
    def comparison_reasoning(self) -> str | None:
        """Convenience property to access comparison_reasoning from the nested answer."""
        if isinstance(self.answer, PairwiseEvaluationAnswer):
            return self.answer.comparison_reasoning
        return None

    @property
    def winner(self) -> Literal["A", "B", "C"] | None:
        """Convenience property to access winner from the nested answer."""
        if isinstance(self.answer, (PairwiseEvaluationAnswer, RubricAnswerFormat)):
            return self.answer.winner
        return None

    def strigify_answer(self) -> str:
        if isinstance(self.answer, PairwiseEvaluationAnswer):
            return f"""Answer A Analysis: {self.answer.answer_a_analysis}
Answer B Analysis: {self.answer.answer_b_analysis}
Comparison Reasoning: {self.answer.comparison_reasoning}
Winner: {self.answer.winner}"""
        if isinstance(self.answer, RubricAnswerFormat):
            return f"""Winner: {self.answer.winner}"""
        return "No answer available"

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


class MetricComparison(BaseModel):
    """Paired comparison of two agents on one metric. Deltas, wins and losses read as agent_b minus agent_a."""

    mean_a: float
    mean_b: float
    p_value: float
    per_query_delta: dict[str, float]

    @computed_field
    @property
    def delta(self) -> float:
        return self.mean_b - self.mean_a

    @computed_field
    @property
    def wins(self) -> int:
        return sum(1 for delta in self.per_query_delta.values() if delta > 0)

    @computed_field
    @property
    def ties(self) -> int:
        return sum(1 for delta in self.per_query_delta.values() if delta == 0)

    @computed_field
    @property
    def losses(self) -> int:
        return sum(1 for delta in self.per_query_delta.values() if delta < 0)


class RetrievalComparisonResult(BaseModel):
    agent_a: str
    agent_b: str
    metrics: dict[str, MetricComparison]


class RDNAMEvaluatorResult(RetrievalEvaluatorResult):
    """Specialized retrieval result for RDNAM (answer is typically RDNAMEvaluationAnswer)."""

    answer: RDNAMEvaluationAnswer = Field(...)

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        return _resolve_legacy_answer(data, (RDNAMEvaluationAnswer,))

    @property
    def intent_match(self) -> float | None:
        """Convenience property to access intent_match from the nested answer."""
        if self.answer and hasattr(self.answer, "intent_match"):
            return self.answer.intent_match  # type: ignore
        return None

    @property
    def trustworthiness(self) -> float | None:
        """Convenience property to access trustworthiness from the nested answer."""
        if self.answer and hasattr(self.answer, "trustworthiness"):
            return self.answer.trustworthiness  # type: ignore
        return None

    def strigify_answer(self) -> str:
        if (
            self.answer
            and hasattr(self.answer, "score")
            and hasattr(self.answer, "reasoning")
            and hasattr(self.answer, "intent_match")
            and hasattr(self.answer, "trustworthiness")
        ):
            return f"Score: {self.answer.score}\nReasoning: {self.answer.reasoning}\
            Intent Match: {self.answer.intent_match}\nTrustworthiness: {self.answer.trustworthiness}"
        return "No answer available"


class RDNAMNoAspectsResult(RetrievalEvaluatorResult):
    """RDNAM result without aspects (answer is typically RDNAMNoAspectsAnswer)."""

    answer: RDNAMNoAspectsAnswer

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        return _resolve_legacy_answer(data, (RDNAMNoAspectsAnswer,))

    @property
    def score(self) -> float | None:
        """Convenience property to access score score from the nested answer."""
        if self.answer and hasattr(self.answer, "score"):
            return self.answer.score  # type: ignore
        return None


class RDNAMMUltipleAnnotatorsResult(RetrievalEvaluatorResult):
    """RDNAM result simulating multiple annotators (answer is typically RDNAMMultipleAnnotatorsAnswer)."""

    answer: RDNAMMultipleAnnotatorsAnswer

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        return _resolve_legacy_answer(data, (RDNAMMultipleAnnotatorsAnswer,))


class RDNAMMultipleAnnotatorsNoAspectsResult(RetrievalEvaluatorResult):
    """
    RDNAM result simulating multiple annotators without aspects (typically RDNAMMultipleAnnotatorsNoAspectsAnswer).
    """

    answer: RDNAMMultipleAnnotatorsNoAspectsAnswer

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        return _resolve_legacy_answer(data, (RDNAMMultipleAnnotatorsNoAspectsAnswer,))


T_Result = TypeVar("T_Result", bound=EvaluatorResult)
