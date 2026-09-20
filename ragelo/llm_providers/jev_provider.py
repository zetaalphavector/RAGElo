from __future__ import annotations

from typing import Any

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import LLMInputPrompt


class JevProvider(BaseLLMProvider):
    """TypeSafe's Jev answers typed questions about a state with probabilities and generates no text, so only
    the jev evaluators can use it. The subclasses differ in how they reach it, and all return a `JevResponse`.
    """

    @staticmethod
    def _state(input: LLMInputPrompt) -> str:
        if input.user_message is None:
            raise ValueError("Jev judges a state, which the jev evaluators pass as the user message.")
        return input.user_message

    @staticmethod
    def _questions(input: LLMInputPrompt) -> dict[str, dict[str, Any]]:
        """The system prompt is the instruction of every question that does not bring its own."""
        if not input.questions:
            raise ValueError("Jev needs typed questions. Use it with one of the jev evaluators.")
        return {name: {"instructions": input.system_prompt, **question} for name, question in input.questions.items()}
