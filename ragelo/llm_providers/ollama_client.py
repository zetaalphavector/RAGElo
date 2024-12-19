from __future__ import annotations

import json
from typing import Any, Type

from openai import AsyncOpenAI
from pydantic import BaseModel as PydanticBaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.types.configurations import OllamaConfiguration
from ragelo.types.formats import AnswerFormat, LLMResponseType
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
        prompt: str | list[dict[str, str]],
        answer_format: AnswerFormat = AnswerFormat.TEXT,
        response_schema: Type[PydanticBaseModel] | dict[str, Any] | None = None,
    ) -> LLMResponseType:
        """Calls the Ollama Local API asynchronously.

        Args:
            prompt: The prompt to use. Either a list of messages or a string.
        """
        if isinstance(prompt, str):
            prompt = [{"role": "system", "content": prompt}]
        if answer_format == AnswerFormat.STRUCTURED:
            raise NotImplementedError("Structured answer format is not supported on Ollama.")
        if answer_format == AnswerFormat.JSON:
            if isinstance(response_schema, dict):
                schema = json.dumps(response_schema, indent=4)
                prompt[0]["content"] += (
                    "\n\nYour output should be a JSON string that STRICTLY "
                    f"adheres to the following schema:\n{schema}"
                )
            answers = await self.__ollama_client.chat.completions.create(
                model=self.config.model,
                messages=prompt,  # type: ignore
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},
                seed=self.config.seed,
            )
        else:
            answers = await self.__ollama_client.chat.completions.create(
                model=self.config.model,
                messages=prompt,  # type: ignore
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                seed=self.config.seed,
            )
        if not answers.choices or not answers.choices[0].message or not answers.choices[0].message.content:
            raise ValueError("Ollama did not return any completions.")
        if answer_format == AnswerFormat.JSON:
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
