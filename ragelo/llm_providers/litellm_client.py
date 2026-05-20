from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import before_sleep_log, retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.types import LLMInputPrompt, LLMResponseType
from ragelo.types.configurations import LiteLLMConfiguration
from ragelo.types.types import LLMProviderTypes

logger = logging.getLogger(__name__)

T_Schema = TypeVar("T_Schema", bound=BaseModel)


def _is_transient_error(exc: BaseException) -> bool:
    qualname = f"{type(exc).__module__}.{type(exc).__qualname__}"
    return qualname in {
        "litellm.exceptions.RateLimitError",
        "litellm.exceptions.Timeout",
        "litellm.exceptions.APIConnectionError",
        "litellm.exceptions.InternalServerError",
        "litellm.exceptions.ServiceUnavailableError",
    }


@LLMProviderFactory.register(LLMProviderTypes.LITELLM)
class LiteLLMProvider(BaseLLMProvider):
    """LLM provider backed by LiteLLM, supporting 100+ providers through a unified interface.

    Uses provider-prefixed model strings (e.g. ``openai/gpt-4o``,
    ``anthropic/claude-sonnet-4-6``, ``groq/llama-3.3-70b-versatile``).
    API keys are read from environment variables automatically by LiteLLM.

    Requires ``pip install 'ragelo[litellm]'``.
    """

    config: LiteLLMConfiguration
    api_key_env_var: str = ""

    def __init__(self, config: LiteLLMConfiguration) -> None:
        try:
            import litellm  # noqa: F401
        except ImportError:
            raise ImportError("litellm is not installed. Install it with: pip install 'ragelo[litellm]'")
        super().__init__(config)

    @retry(
        retry=retry_if_exception(_is_transient_error),
        wait=wait_random_exponential(min=1, max=120),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger=logger, log_level=logging.INFO),
        reraise=True,
    )
    async def call_async(self, input: LLMInputPrompt, response_schema: type[T_Schema]) -> LLMResponseType[T_Schema]:
        """Calls the LLM via LiteLLM asynchronously.

        Args:
            input: A LLMInputPrompt object containing the system prompt, user message, or a list of messages.
            response_schema: The Pydantic schema the LLM should fill in.

        Returns:
            LLMResponseType with raw_answer (str) and parsed_answer (instance of response_schema).
        """
        import litellm

        messages: list[dict[str, str]] = []
        if input.messages:
            messages = input.messages
        else:
            if input.system_prompt:
                messages.append({"role": "system", "content": input.system_prompt})
            if input.user_message:
                messages.append({"role": "user", "content": input.user_message})
        if not messages:
            raise ValueError("No input provided")

        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            schema_dict = response_schema.model_json_schema()
        else:
            schema_dict = response_schema  # type: ignore

        schema = json.dumps(schema_dict, indent=4)
        messages[-1]["content"] += (
            f"\n\nYour output should be a JSON string that STRICTLY adheres to the following schema:\n{schema}"
        )

        call_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "drop_params": True,
        }
        if self.config.api_key:
            call_kwargs["api_key"] = self.config.api_key.get_secret_value()
        if self.config.api_base:
            call_kwargs["api_base"] = self.config.api_base

        response = await litellm.acompletion(**call_kwargs)

        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            raise ValueError("LiteLLM did not return any completions.")

        raw_answer = response.choices[0].message.content
        try:
            parsed_answer = response_schema.model_validate_json(raw_answer)
        except ValidationError as e:
            raise ValueError(
                f"Failed to parse raw JSON answer {raw_answer} into the response schema {response_schema}: {e}"
            ) from e

        return LLMResponseType(
            raw_answer=raw_answer,
            parsed_answer=parsed_answer,
        )
