from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.types import LLMInputPrompt, LLMResponseType
from ragelo.types.configurations import OllamaConfiguration
from ragelo.types.types import LLMProviderTypes

logger = logging.getLogger(__name__)

T_Schema = TypeVar("T_Schema", bound=BaseModel)


@LLMProviderFactory.register(LLMProviderTypes.OLLAMA)
class OllamaProvider(BaseLLMProvider):
    """A Wrapper over the Ollama client."""

    config: OllamaConfiguration

    def __init__(
        self,
        config: OllamaConfiguration,
    ):
        super().__init__(config)
        self.__ollama_client = self.__get_ollama_client(config)

    @retry(
        wait=wait_random_exponential(min=1, max=120),
        stop=stop_after_attempt(1),
        before_sleep=before_sleep_log(logger=logger, log_level=logging.INFO),
    )
    async def call_async(self, input: LLMInputPrompt, response_schema: type[BaseModel]) -> LLMResponseType:
        """Calls the Ollama Local API asynchronously.

        Args:
            input: A LLMInputPrompt object containing the system prompt, user message, or a list of messages.
            response_schema: The schema of the response to expect. If config.json_mode is True,
                we will dump the response_schema to a JSON string and add it to the instructions.
        Returns:
            The response from the Ollama API, formatted according to the response_schema.
                The LLMResponseType.raw_answer contains the raw LLM response as a string and the
                LLMResponseType.parsed_answer contains the parsed response.
                This can be a number or string (if response_schema is None) a dictionary
                (if response_schema is a dictionary) or a Pydantic BaseModel
                (if response_schema is a Pydantic BaseModel)).
        """
        messages = []
        call_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
        }

        if input.system_prompt and input.messages:
            raise ValueError(
                "If the input to the LLMProvider is a list of messages, you should not provide the system prompt as a "
                "unique parameter. Please combine both in a single input list"
            )
        if input.system_prompt:
            messages.append({"role": "system", "content": input.system_prompt})
        if input.user_message:
            messages.append({"role": "user", "content": input.user_message})
        if input.messages:
            messages = input.messages
        if not messages:
            raise ValueError("No input provided")

        if self.config.reasoning_effort:
            call_kwargs["extra_body"] = {"reasoning_effort": self.config.reasoning_effort}

        call_kwargs["messages"] = messages
        if self.config.json_mode:
            # Build a JSON schema from a Pydantic model class when available
            if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                schema_dict = response_schema.model_json_schema()
            else:
                schema_dict = response_schema  # type: ignore
            schema = json.dumps(schema_dict, indent=4)
            messages[-1]["content"] += (
                f"\n\nYour output should be a JSON string that STRICTLY adheres to the following schema:\n{schema}"
            )
            call_kwargs["response_format"] = {"type": "json_object"}
            answers = await self.__ollama_client.chat.completions.create(**call_kwargs)  # type: ignore
            if not answers.choices or not answers.choices[0].message or not answers.choices[0].message.content:
                raise ValueError("Ollama did not return any completions.")
            parsed_answer = json.loads(answers.choices[0].message.content)
            raw_answer = answers.choices[0].message.content
            try:
                parsed_answer = response_schema.model_validate_json(parsed_answer)
            except ValidationError as e:
                raise ValueError(
                    f"Failed to parse raw JSON answer {raw_answer} into the response schema {response_schema}: {e}"
                ) from e

        else:
            call_kwargs["response_format"] = response_schema
            answers = await self.__ollama_client.chat.completions.parse(**call_kwargs)  # type: ignore
            if not answers.choices or not answers.choices[0].message or not answers.choices[0].message.content:
                raise ValueError("Ollama did not return any completions.")
            parsed_answer = answers.choices[0].message.parsed
            raw_answer = answers.choices[0].message.content

        return LLMResponseType(
            raw_answer=raw_answer,
            parsed_answer=parsed_answer,
        )

    @staticmethod
    def __get_ollama_client(ollama_config: OllamaConfiguration) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=ollama_config.api_base,
            api_key=ollama_config.api_key,
        )
