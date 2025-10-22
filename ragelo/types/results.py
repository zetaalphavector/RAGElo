from __future__ import annotations

from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, Field, computed_field, field_serializer, model_validator
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
)


class EvaluatorResult(BaseModel):
    """Generic class with the results of an evaluator.
    Args:
        qid str: The query ID to which the result corresponds.
        evaluator_name str: The name of the evaluator that produced this result.
        exception Optional[str]: Any exception captured during evaluation.
        answer: The LLM-generated evaluation content (without metadata).
    """

    qid: Annotated[str, SkipJsonSchema] = Field(description="The query ID to which the result corresponds.")
    evaluator_name: Annotated[str, SkipJsonSchema] = Field(
        description="The name of the evaluator that produced this result."
    )
    exception: Annotated[str | None, SkipJsonSchema] = Field(
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

    did: Annotated[str, SkipJsonSchema] = Field(description="The document ID to which the result corresponds.")
    answer: RetrievalEvaluationAnswer

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        """Ensure answer is deserialized as the correct subclass type."""
        if isinstance(data, dict) and "answer" in data and data["answer"] is not None:
            answer_data = data["answer"]
            if isinstance(answer_data, dict) and not isinstance(answer_data, BaseModel):
                # Deserialize as RetrievalEvaluationAnswer by default
                data["answer"] = RetrievalEvaluationAnswer.model_validate(answer_data)
        return data

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

    agent: Annotated[str, SkipJsonSchema] = Field(description="The agent that provided the answer.")
    answer: AnswerEvaluationAnswer

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        """Ensure answer is deserialized as the correct subclass type."""
        if isinstance(data, dict) and "answer" in data and data["answer"] is not None:
            answer_data = data["answer"]
            if isinstance(answer_data, dict) and not isinstance(answer_data, BaseModel):
                # Deserialize as AnswerEvaluationAnswer by default
                data["answer"] = AnswerEvaluationAnswer.model_validate(answer_data)
        return data

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

    agent_a: Annotated[str, SkipJsonSchema]
    agent_b: Annotated[str, SkipJsonSchema]
    answer: PairwiseEvaluationAnswer

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        """Ensure answer is deserialized as the correct subclass type."""
        if isinstance(data, dict) and "answer" in data and data["answer"] is not None:
            answer_data = data["answer"]
            if isinstance(answer_data, dict) and not isinstance(answer_data, BaseModel):
                # Deserialize as PairwiseEvaluationAnswer by default
                data["answer"] = PairwiseEvaluationAnswer.model_validate(answer_data)
        return data

    @property
    def answer_a_analysis(self) -> str | None:
        """Convenience property to access answer_a_analysis from the nested answer."""
        if self.answer and hasattr(self.answer, "answer_a_analysis"):
            return self.answer.answer_a_analysis  # type: ignore
        return None

    @property
    def answer_b_analysis(self) -> str | None:
        """Convenience property to access answer_b_analysis from the nested answer."""
        if self.answer and hasattr(self.answer, "answer_b_analysis"):
            return self.answer.answer_b_analysis  # type: ignore
        return None

    @property
    def comparison_reasoning(self) -> str | None:
        """Convenience property to access comparison_reasoning from the nested answer."""
        if self.answer and hasattr(self.answer, "comparison_reasoning"):
            return self.answer.comparison_reasoning  # type: ignore
        return None

    @property
    def winner(self) -> Literal["A", "B", "C"] | None:
        """Convenience property to access winner from the nested answer."""
        if self.answer and hasattr(self.answer, "winner"):
            return self.answer.winner  # type: ignore
        return None

    def strigify_answer(self) -> str:
        if (
            self.answer
            and hasattr(self.answer, "answer_a_analysis")
            and hasattr(self.answer, "answer_b_analysis")
            and hasattr(self.answer, "comparison_reasoning")
            and hasattr(self.answer, "winner")
        ):
            return f"""Answer A Analysis: {self.answer.answer_a_analysis}
Answer B Analysis: {self.answer.answer_b_analysis}
Comparison Reasoning: {self.answer.comparison_reasoning}
Winner: {self.answer.winner}"""
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


class RDNAMEvaluatorResult(RetrievalEvaluatorResult):
    """Specialized retrieval result for RDNAM (answer is typically RDNAMEvaluationAnswer)."""

    answer: RDNAMEvaluationAnswer

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        """Ensure answer is deserialized as RDNAMEvaluationAnswer."""
        if isinstance(data, dict) and "answer" in data and data["answer"] is not None:
            answer_data = data["answer"]
            if isinstance(answer_data, dict) and not isinstance(answer_data, BaseModel):
                # Deserialize as RDNAMEvaluationAnswer
                data["answer"] = RDNAMEvaluationAnswer.model_validate(answer_data)
        return data

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
            return f"Score: {self.answer.score}\nReasoning: {self.answer.reasoning}\nIntent Match: {self.answer.intent_match}\nTrustworthiness: {self.answer.trustworthiness}"
        return "No answer available"


class RDNAMNoAspectsResult(RetrievalEvaluatorResult):
    """RDNAM result without aspects (answer is typically RDNAMNoAspectsAnswer)."""

    answer: RDNAMNoAspectsAnswer

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        """Ensure answer is deserialized as RDNAMNoAspectsAnswer."""
        if isinstance(data, dict) and "answer" in data and data["answer"] is not None:
            answer_data = data["answer"]
            if isinstance(answer_data, dict) and not isinstance(answer_data, BaseModel):
                data["answer"] = RDNAMNoAspectsAnswer.model_validate(answer_data)
        return data

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
        """Ensure answer is deserialized as RDNAMMultipleAnnotatorsAnswer."""
        if isinstance(data, dict) and "answer" in data and data["answer"] is not None:
            answer_data = data["answer"]
            if isinstance(answer_data, dict) and not isinstance(answer_data, BaseModel):
                data["answer"] = RDNAMMultipleAnnotatorsAnswer.model_validate(answer_data)
        return data


class RDNAMMultipleAnnotatorsNoAspectsResult(RetrievalEvaluatorResult):
    """RDNAM result simulating multiple annotators without aspects (answer is typically RDNAMMultipleAnnotatorsNoAspectsAnswer)."""

    answer: RDNAMMultipleAnnotatorsNoAspectsAnswer

    @model_validator(mode="before")
    @classmethod
    def validate_answer_type(cls, data: Any) -> Any:
        """Ensure answer is deserialized as RDNAMMultipleAnnotatorsNoAspectsAnswer."""
        if isinstance(data, dict) and "answer" in data and data["answer"] is not None:
            answer_data = data["answer"]
            if isinstance(answer_data, dict) and not isinstance(answer_data, BaseModel):
                data["answer"] = RDNAMMultipleAnnotatorsNoAspectsAnswer.model_validate(answer_data)
        return data


T_Result = TypeVar("T_Result", bound=EvaluatorResult)
