from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator

from ragelo.logger import logger


class LLMInputPrompt(BaseModel):
    system_prompt: str | None = None
    user_message: str | None = None
    messages: list[dict[str, str]] | None = None

    @model_validator(mode="after")
    def check_at_least_one_is_set(self) -> "LLMInputPrompt":
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


class LLMResponseType(BaseModel):
    raw_answer: str
    parsed_answer: float | str | dict[str, Any] | BaseModel | None
