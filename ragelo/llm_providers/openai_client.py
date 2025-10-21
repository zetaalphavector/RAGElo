from __future__ import annotations

import json
from typing import Any, Type

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.responses import ResponseTextConfigParam
from openai.types.shared_params import ResponseFormatJSONObject
from pydantic import BaseModel, ValidationError
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
        if self.config.model.startswith("gpt-5") or self.config.model.startswith("o"):
            self.config.temperature = None

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(3))
    async def call_async(self, input: LLMInputPrompt, response_schema: Type[BaseModel]) -> LLMResponseType:
        """Calls the OpenAI API asynchronously.

        Args:
            input: A LLMInputPrompt object containing the system prompt, user message, or a list of messages.
            response_schema: The schema of the response to expect. If config.json_mode is True, we will dump the response_schema to a JSON string and add it to the instructions.
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

        if self.config.json_mode:
            # Build a JSON schema from a Pydantic model class when available
            if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                schema_dict = response_schema.model_json_schema()
            else:
                # Fallback for dict-like schemas provided directly
                schema_dict = response_schema  # type: ignore
            schema = json.dumps(schema_dict, indent=4)
            if isinstance(llm_input, str):
                llm_input += (
                    f"\n\nYour output should be a JSON string that STRICTLY adheres to the following schema:\n{schema}"
                )
            else:
                llm_input[-1]["content"] += (
                    f"\n\nYour output should be a JSON string that STRICTLY adheres to the following schema:\n{schema}"
                )
            answers = await self.__openai_client.responses.create(
                input=llm_input,  # type: ignore
                instructions=instructions,
                model=self.config.model,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                text=ResponseTextConfigParam(format=ResponseFormatJSONObject(type="json_object")),
            )
            raw_answer = answers.output_text
            parsed_answer = json.loads(raw_answer)
            try:
                parsed_answer = response_schema.model_validate_json(parsed_answer)
            except ValidationError as e:
                raise ValueError(
                    f"Failed to parse raw JSON answer {raw_answer} into the response schema {response_schema}: {e}"
                ) from e

        else:
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
