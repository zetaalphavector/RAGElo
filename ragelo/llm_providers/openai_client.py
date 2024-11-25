from __future__ import annotations

import json
from typing import Any, Type

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai._types import NOT_GIVEN
from pydantic import BaseModel as PydanticBaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.logger import logger
from ragelo.types.configurations import OpenAIConfiguration
from ragelo.types.formats import AnswerFormat
from ragelo.types.pydantic_models import _PYDANTIC_MAJOR_VERSION
from ragelo.types.types import LLMProviderTypes
from ragelo.utils import call_async_fn


@LLMProviderFactory.register(LLMProviderTypes.OPENAI)
class OpenAIProvider(BaseLLMProvider):
    """A Wrapper over the OpenAI client."""

    config: OpenAIConfiguration
    api_key_env_var: str = "OPENAI_API_KEY"

    def __init__(
        self,
        config: OpenAIConfiguration,
    ):
        super().__init__(config)
        self.__openai_client = self.__get_openai_client(config)

    def __call__(
        self,
        prompt: str | list[dict[str, str]],
        answer_format: AnswerFormat = AnswerFormat.TEXT,
        response_schema: Type[PydanticBaseModel] | dict[str, Any] | None = None,
    ) -> str | PydanticBaseModel | dict[str, Any]:
        """Calls the OpenAI API.

        Args:
            prompt: The prompt to use. Either a list of messages or a string.
            answer_format: The format of the answer to return. Either TEXT, JSON, or STRUCTURED.
            response_format: The format of the response to expect. If the answer_format is STRUCTURED,
                this should be a PydanticBaseModel class. If the answer_format is JSON, this should be a dictionary
                with the desired format the answer should be in
        Returns:
            The response from the OpenAI API, formatted according to the answer_format.
        """
        return call_async_fn(self.call_async, prompt, answer_format, response_schema)

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(1))
    async def call_async(
        self,
        prompt: str | list[dict[str, str]],
        answer_format: AnswerFormat = AnswerFormat.TEXT,
        response_schema: Type[PydanticBaseModel] | dict[str, Any] | None = None,
    ) -> str | PydanticBaseModel | dict[str, Any]:
        """Calls the OpenAI API asynchronously.

        Args:
            prompt: The prompt to use. Either a list of messages or a string.
            answer_format: The format of the answer to return. Either TEXT, JSON, or STRUCTURED.
            response_format: The format of the response to expect. If the answer_format is STRUCTURED,
                this should be a PydanticBaseModel class. If the answer_format is JSON, this should be a dictionary
                with the desired format the answer should be in
        Returns:
            The response from the OpenAI API, formatted according to the answer_format. or a string.
        """
        if isinstance(prompt, str):
            prompt = [{"role": "system", "content": prompt}]

        if answer_format == AnswerFormat.STRUCTURED:
            if not isinstance(response_schema, type(PydanticBaseModel)):
                raise ValueError("response_schema must be a PydanticBaseModel Class when using structured output.")
            answers = await self.__openai_client.beta.chat.completions.parse(
                messages=prompt,  # type: ignore
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format=response_schema,
            )
        else:
            if answer_format == AnswerFormat.JSON:
                if isinstance(response_schema, type(PydanticBaseModel)):
                    logger.info(
                        "You provided a PydanticBaseModel class for the response_schema. "
                        "Using JSON as the desired answer format. Dumping the schema to JSON."
                    )
                    if _PYDANTIC_MAJOR_VERSION >= 2:
                        response_schema = response_schema.model_json_schema()
                    else:
                        response_schema = response_schema.schema()  # type: ignore
                elif isinstance(response_schema, dict):
                    schema = json.dumps(response_schema, indent=4)
                    prompt[0]["content"] += (
                        "\n\nYour output should be a JSON string that STRICTLY "
                        f"adheres to the following schema:\n{schema}"
                    )
                response_format_parameter = {"type": "json_object"}
            else:
                response_format_parameter = NOT_GIVEN  # type: ignore
            answers = await self.__openai_client.chat.completions.create(
                messages=prompt,  # type: ignore
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format=response_format_parameter,  # type: ignore
            )

        if not answers.choices or not answers.choices[0].message or not answers.choices[0].message.content:
            raise ValueError("OpenAI did not return any completions.")
        if answer_format == AnswerFormat.STRUCTURED:
            return answers.choices[0].message.parsed  # type: ignore
        if answer_format == AnswerFormat.JSON:
            return json.loads(answers.choices[0].message.content)
        return answers.choices[0].message.content

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
