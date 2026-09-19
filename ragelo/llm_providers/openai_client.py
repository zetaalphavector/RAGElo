from __future__ import annotations

import json
from typing import Any, TypeVar

from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.responses import ResponseFormatTextJSONSchemaConfigParam, ResponseTextConfigParam
from pydantic import BaseModel, ValidationError

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.types import LLMInputPrompt, LLMResponseType
from ragelo.types.configurations import OpenAIConfiguration
from ragelo.types.types import LLMProviderTypes

T_Schema = TypeVar("T_Schema", bound=BaseModel)


@LLMProviderFactory.register(LLMProviderTypes.OPENAI)
class OpenAIProvider(BaseLLMProvider):
    """A Wrapper over the OpenAI client."""

    config: OpenAIConfiguration
    api_key_env_var: str = "OPENAI_API_KEY"

    def __init__(self, config: OpenAIConfiguration, client: AsyncOpenAI | None = None) -> None:
        super().__init__(config)
        self.__openai_client = client or self.__get_openai_client(config)
        creator, _, model = self.config.model.rpartition("/")
        if creator in ("", "openai"):
            if model.startswith(("gpt-5", "o")):
                self.config.temperature = None
            elif self.config.reasoning_effort:
                self.config.reasoning_effort = None

    async def call_async(self, input: LLMInputPrompt, response_schema: type[T_Schema]) -> LLMResponseType[T_Schema]:
        """Calls the OpenAI API asynchronously.

        Args:
            input: A LLMInputPrompt object containing the system prompt, user message, or a list of messages.
            response_schema: The schema of the response (typically an EvaluationAnswer subclass like
                RetrievalEvaluationAnswer or AnswerEvaluationAnswer). This is the schema the LLM should fill in.
        Returns:
            The response from the OpenAI Responses API. The LLMResponseType.raw_answer contains the raw LLM response
            as a string and the LLMResponseType.parsed_answer contains the parsed response as an instance
            of response_schema.
        """
        llm_input: str | list[dict[str, str]]
        if input.messages:
            llm_input = input.messages
        elif input.user_message:
            llm_input = input.user_message
        else:
            raise ValueError("No input provided")
        optional_kwargs = {
            "instructions": input.system_prompt,
            "temperature": self.config.temperature,
            "reasoning": {"effort": self.config.reasoning_effort} if self.config.reasoning_effort else None,
        }
        call_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_output_tokens": self.config.max_tokens,
            **{k: v for k, v in optional_kwargs.items() if v is not None},
        }

        if self.config.json_mode:
            schema_dict = response_schema.model_json_schema()
            schema = json.dumps(schema_dict, indent=4)
            suffix = (
                f"\n\nYour output should be a JSON string that STRICTLY adheres to the following schema:\n{schema}"
            )
            if isinstance(llm_input, str):
                llm_input += suffix
            else:
                last = llm_input[-1]
                llm_input = [*llm_input[:-1], {**last, "content": last["content"] + suffix}]
            try:
                answer = await self.__openai_client.responses.create(
                    input=llm_input,  # type: ignore
                    text=ResponseTextConfigParam(
                        format=ResponseFormatTextJSONSchemaConfigParam(
                            name=schema_dict.get("title", "response"),
                            schema=schema_dict,
                            type="json_schema",
                        )
                    ),
                    **call_kwargs,
                )
            except Exception as e:
                raise ValueError(f"OpenAI request failed: {e}") from e
            raw_answer = answer.output_text
            try:
                parsed_answer = response_schema.model_validate_json(raw_answer)
            except ValidationError as e:
                raise ValueError(
                    f"Failed to parse raw JSON answer {raw_answer} into the response schema {response_schema}: {e}"
                ) from e

        else:
            try:
                answer = await self.__openai_client.responses.parse(
                    text_format=response_schema,
                    input=llm_input,  # type: ignore
                    **call_kwargs,
                )
            except Exception as e:
                raise ValueError(f"OpenAI request failed: {e}") from e
            if not isinstance(answer.output_parsed, response_schema):
                raise ValueError(
                    f"OpenAI failed to parse response into the response schema {response_schema}. "
                    f"The response was: {answer.output_text}"
                )
            parsed_answer = answer.output_parsed
            raw_answer = answer.output_text

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
                api_key=openai_config.api_key.get_secret_value(),
                api_version=openai_config.api_version,
            )
        elif openai_config.api_type in ("openai", "open_ai", None):
            return AsyncOpenAI(
                base_url=openai_config.api_base,
                api_key=openai_config.api_key.get_secret_value(),
                organization=openai_config.org,
            )
        else:
            raise ValueError(f"Unknown OpenAI api type: {openai_config.api_type}")
