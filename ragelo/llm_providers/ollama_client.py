from __future__ import annotations

import json
from typing import Any, Type

from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.types.configurations import OllamaConfiguration
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.types import LLMProviderTypes


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

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(1))
    async def call_async(
        self,
        input: LLMInputPrompt,
        response_schema: Type[BaseModel] | dict[str, Any] | None = None,
    ) -> LLMResponseType:
        """Calls the Ollama Local API asynchronously.

        Args:
            user_prompt: The user prompt to send to the model.
            system_prompt: The system prompt to send to the model.
            answer_format: The format of the answer to return.
            response_schema: The response schema for structured output.
        """
        messages = []
        call_kwargs = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
        }

        if input.system_prompt and input.messages:
            raise ValueError(
                "If the input to the LLMProvider is a list of messages, you should not provide the system prompt as a unique parameter. Please combine both in a single input list"
            )
        if input.system_prompt:
            messages.append({"role": "system", "content": input.system_prompt})
        if input.user_message:
            messages.append({"role": "user", "content": input.user_message})
        if input.messages:
            messages = input.messages
        if not messages:
            raise ValueError("No input provided")

        call_kwargs["messages"] = messages
        if isinstance(response_schema, type(BaseModel)):
            raise NotImplementedError("Structured answer format is not supported on Ollama.")
        if isinstance(response_schema, dict):
            schema = json.dumps(response_schema, indent=4)
            messages[-1]["content"] += (
                f"\n\nYour output should be a JSON string that STRICTLY adheres to the following schema:\n{schema}"
            )
            call_kwargs["response_format"] = {"type": "json_object"}
        answers = await self.__ollama_client.chat.completions.create(**call_kwargs)  # type: ignore

        if not answers.choices or not answers.choices[0].message or not answers.choices[0].message.content:
            raise ValueError("Ollama did not return any completions.")

        if isinstance(response_schema, dict):
            return LLMResponseType(
                raw_answer=answers.choices[0].message.content,
                parsed_answer=json.loads(answers.choices[0].message.content),
            )
        return LLMResponseType(
            raw_answer=answers.choices[0].message.content,
            parsed_answer=answers.choices[0].message.content,
        )

    @staticmethod
    def __get_ollama_client(ollama_config: OllamaConfiguration) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=ollama_config.api_base,
            api_key=ollama_config.api_key,
        )
