from __future__ import annotations

import json
from typing import Any, Type

from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.types.configurations import OpenAIConfiguration
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.types import LLMProviderTypes


@LLMProviderFactory.register(LLMProviderTypes.OPENAI)
class OpenAIProvider(BaseLLMProvider):
    """A Wrapper over the OpenAI client."""

    config: OpenAIConfiguration
    api_key_env_var: str = "OPENAI_API_KEY"

    def __init__(self, config: OpenAIConfiguration) -> None:
        super().__init__(config)
        self.__openai_client = self.__get_openai_client(config)

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(5))
    async def call_async(
        self,
        input: LLMInputPrompt,
        response_schema: Type[BaseModel] | dict[str, Any] | None = None,
    ) -> LLMResponseType:
        """Calls the OpenAI API asynchronously.

        Args:
            input: A LLMInputPrompt object containing the system prompt, user message, or a list of messages.
            response_schema: The schema of the response to expect. If the answer_format is STRUCTURED,
                this should be a Pydantic BaseModel class (not instance). If the answer_format is JSON, this should be
                a dictionary with the desired format the answer should be in. Otherwise, the response will be returned as a string.
        Returns:
            The response from the OpenAI Responses API, formatted according to the answer_format. The LLMResponseType.raw_answer  contains the raw LLM response as a string and the LLMResponseType.parsed_answer contains the parsed response. This can be a number or string (if response_schema is None) a dictionary (if response_schema is a dictionary) or a Pydantic BaseModel (if response_schema is a Pydantic BaseModel)).
        """
        llm_input: str | list[dict[str, str]]
        if input.messages:
            llm_input = input.messages
        elif input.user_message:
            llm_input = input.user_message
        else:
            raise ValueError("No input provided")
        parsed_answer: str | dict[str, Any] | BaseModel | None = None
        if input.system_prompt:
            instructions = input.system_prompt
        else:
            instructions = None
        if isinstance(response_schema, type(BaseModel)):
            answers = await self.__openai_client.responses.parse(
                text_format=response_schema,
                input=llm_input,  # type: ignore
                instructions=instructions,
                model=self.config.model,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
            parsed_answer = answers.output_parsed
            raw_answer = answers.output_text
        elif isinstance(response_schema, dict):
            llm_text = {
                "format": {
                    "type": "json_schema",
                    "schema": response_schema,
                    "strict": True,
                }
            }
            answers = await self.__openai_client.responses.create(
                input=llm_input,  # type: ignore
                instructions=instructions,
                model=self.config.model,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                text=llm_text,
            )
            raw_answer = answers.output_text
            try:
                parsed_answer = json.loads(raw_answer)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse raw JSON answer {raw_answer} as JSON: {e}") from e
        else:
            answers = await self.__openai_client.responses.create(
                input=llm_input,  # type: ignore
                instructions=instructions,
                model=self.config.model,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
            raw_answer = answers.output_text
            parsed_answer = raw_answer

        return LLMResponseType(
            raw_answer=raw_answer,
            parsed_answer=parsed_answer,
        )

    @staticmethod
    def __get_openai_client(openai_config: OpenAIConfiguration) -> AsyncOpenAI:
        if openai_config.api_type == "azure":
            if openai_config.api_base is None:
                raise ValueError("Azure-OpenAI base url (api_base) not found in configuration.")
            return AsyncAzureOpenAI(
                azure_endpoint=openai_config.api_base,
                api_key=openai_config.api_key,
                api_version=openai_config.api_version,
            )
        elif openai_config.api_type == "open_ai" or openai_config.api_type is None:
            return AsyncOpenAI(
                base_url=openai_config.api_base,
                api_key=openai_config.api_key,
                organization=openai_config.org,
            )
        else:
            raise Exception(f"Unknown OpenAI api type: {openai_config.api_type}")
