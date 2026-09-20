from __future__ import annotations

import logging
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, model_validator

logger = logging.getLogger(__name__)

T_Schema = TypeVar("T_Schema", bound=BaseModel)


class LLMInputPrompt(BaseModel):
    system_prompt: str | None = None
    user_message: str | None = None
    messages: list[dict[str, str]] | None = None
    llm_response_schema: type[BaseModel] | dict[str, Any] | None = None
    questions: dict[str, dict[str, Any]] | None = None
    batch_key: str | None = None

    @model_validator(mode="after")
    def check_at_least_one_is_set(self) -> LLMInputPrompt:
        if not self.system_prompt and not self.user_message and not self.messages:
            raise ValueError("At least one of system_prompt, user_message, or messages must be set")
        if self.system_prompt and not self.messages and not self.user_message:
            logger.warning(
                "You are providing a system prompt, but no messages. "
                "The system prompt will be set as the user_message."
            )
            self.user_message = self.system_prompt
            self.system_prompt = None
        return self


class LLMUsage(BaseModel):
    """Tokens one LLM call was billed for. `cached_tokens` is the part of `input_tokens` read from a cache."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: LLMUsage) -> LLMUsage:
        return LLMUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


class LLMResponseType(BaseModel, Generic[T_Schema]):
    raw_answer: str
    parsed_answer: T_Schema
    usage: LLMUsage | None = None


class JevAnswer(BaseModel):
    type: Literal["score", "choice", "boolean"]
    score: float | None = None
    probability: float | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None

    @property
    def label(self) -> str:
        """The most likely option of a choice, or level of a score."""
        if not self.probabilities:
            raise ValueError(f"A {self.type} answer has no labels")
        return max(self.probabilities, key=lambda label: self.probabilities[label])  # type: ignore[index]

    @property
    def is_yes(self) -> bool:
        if self.probability is None:
            raise ValueError(f"A {self.type} answer is not a yes/no answer")
        return self.probability >= 0.5


class JevResponse(BaseModel):
    answers: dict[str, JevAnswer]
