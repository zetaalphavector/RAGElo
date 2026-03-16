from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.types import LLMInputPrompt, LLMResponseType
from ragelo.types.configurations import InstructorConfiguration
from ragelo.types.types import LLMProviderTypes

logger = logging.getLogger(__name__)

T_Schema = TypeVar("T_Schema", bound=BaseModel)

try:
    import instructor
    from instructor import AsyncInstructor
    from instructor.cache import AutoCache
    from instructor.core import ConfigurationError

    _INSTRUCTOR_AVAILABLE = True
except ImportError:
    _INSTRUCTOR_AVAILABLE = False


@LLMProviderFactory.register(LLMProviderTypes.INSTRUCTOR)
class InstructorProvider(BaseLLMProvider):
    """LLM provider backed by the instructor library for structured Pydantic outputs.

    Supports any provider that instructor supports (Anthropic, OpenAI, Mistral, Cohere, etc.)
    through a unified interface. Requires ``pip install 'ragelo[instructor]'`` plus the
    specific provider SDK (e.g. ``pip install anthropic``).
    """

    config: InstructorConfiguration
    api_key_env_var: str = ""

    def __init__(self, config: InstructorConfiguration) -> None:
        if not _INSTRUCTOR_AVAILABLE:
            raise ImportError("instructor is not installed. Install it with: pip install 'ragelo[instructor]'")
        super().__init__(config)
        self.__instructor_client = self.__get_instructor_client(config)

    @retry(
        wait=wait_random_exponential(min=1, max=120),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger=logger, log_level=logging.INFO),
    )
    async def call_async(self, input: LLMInputPrompt, response_schema: type[T_Schema]) -> LLMResponseType[T_Schema]:
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

        call_kwargs: dict[str, Any] = self.config.model_kwargs
        if self.config.temperature is not None:
            call_kwargs["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            call_kwargs["max_tokens"] = self.config.max_tokens
        try:
            parsed_answer = await self.__instructor_client.create(
                response_model=response_schema,
                messages=messages,  # type: ignore
                max_retries=self.config.max_retries,
                **call_kwargs,
            )
        except Exception as e:
            raise ValueError(f"Instructor request failed for Instructor with model '{self.config.model}': {e}") from e

        if not isinstance(parsed_answer, response_schema):
            raise ValueError(
                f"Instructor response could not be parsed into the expected schema {response_schema}. "
                f"Received response: {parsed_answer}"
            )
        raw_answer = parsed_answer.model_dump_json()
        return LLMResponseType(raw_answer=raw_answer, parsed_answer=parsed_answer)

    @staticmethod
    def __get_instructor_client(config: InstructorConfiguration) -> AsyncInstructor:
        try:
            cache = AutoCache(config.cache_size) if config.use_cache else None
            provider_kwargs: dict[str, Any] = {**config.model_kwargs}
            if config.api_key is not None:
                provider_kwargs["api_key"] = config.api_key.get_secret_value()
            instructor_client = instructor.from_provider(
                config.model, async_client=True, cache=cache, **provider_kwargs
            )
        except ConfigurationError as e:
            raise ValueError(f"Failed to initialize instructor client for model '{config.model}': {e}") from e

        return instructor_client
