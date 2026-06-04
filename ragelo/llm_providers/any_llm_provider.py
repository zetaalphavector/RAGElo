import logging
from typing import Any, TypeVar, cast

from any_llm import AnyLLM
from any_llm.types.completion import ChatCompletionMessage
from pydantic import BaseModel
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.types import LLMInputPrompt, LLMResponseType
from ragelo.types.configurations import AnyLLMConfiguration
from ragelo.types.types import LLMProviderTypes

logger = logging.getLogger(__name__)

T_Schema = TypeVar("T_Schema", bound=BaseModel)


@LLMProviderFactory.register(LLMProviderTypes.ANYLLM)
class AnyLLMProvider(BaseLLMProvider):
    """LLM provider backed by Mozilla's any-llm library.

    Supports any provider that any-llm supports through a unified interface.
    """

    config: AnyLLMConfiguration
    api_key_env_var: str = ""

    def __init__(self, config: AnyLLMConfiguration, client: AnyLLM | None = None) -> None:
        super().__init__(config)
        self.__any_llm_client = client or self.__get_any_llm_client(config)

    @retry(
        wait=wait_random_exponential(min=1, max=120),
        stop=stop_after_attempt(3),
        before_sleep=before_sleep_log(logger=logger, log_level=logging.INFO),
    )
    async def call_async(self, input: LLMInputPrompt, response_schema: type[T_Schema]) -> LLMResponseType[T_Schema]:
        messages: list[dict[str, Any] | ChatCompletionMessage] = []
        if input.messages:
            messages = cast(list[dict[str, Any] | ChatCompletionMessage], input.messages)
        else:
            if input.system_prompt:
                messages.append({"role": "system", "content": input.system_prompt})
            if input.user_message:
                messages.append({"role": "user", "content": input.user_message})
        if not messages:
            raise ValueError("No input provided")

        call_kwargs: dict[str, Any] = {**self.config.model_kwargs}
        if self.config.temperature is not None:
            call_kwargs["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            call_kwargs["max_tokens"] = self.config.max_tokens
        if self.config.seed is not None:
            call_kwargs["seed"] = self.config.seed

        try:
            response = await self.__any_llm_client.acompletion(
                model=self.config.model,
                messages=messages,
                response_format=response_schema,
                **call_kwargs,
            )
        except Exception as e:
            raise ValueError(
                f"AnyLLM request failed for provider '{self.config.provider}' with model '{self.config.model}': {e}"
            ) from e

        message = response.choices[0].message
        raw_answer = message.content or ""
        parsed_answer = message.parsed
        if not isinstance(parsed_answer, response_schema):
            parsed_answer = response_schema.model_validate_json(raw_answer)

        return LLMResponseType(raw_answer=raw_answer, parsed_answer=parsed_answer)

    @staticmethod
    def __get_any_llm_client(config: AnyLLMConfiguration) -> AnyLLM:
        client_kwargs: dict[str, Any] = {**config.model_kwargs}
        if config.api_key is not None:
            client_kwargs["api_key"] = config.api_key.get_secret_value()
        if config.api_base is not None:
            client_kwargs["api_base"] = config.api_base
        if config.api_version is not None:
            client_kwargs["api_version"] = config.api_version

        return AnyLLM.create(config.provider, **client_kwargs)
