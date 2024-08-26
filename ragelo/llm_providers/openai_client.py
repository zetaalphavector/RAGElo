import asyncio
from typing import Dict, List, Union

from openai import AsyncAzureOpenAI, AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.types import LLMProviderTypes
from ragelo.types.configurations import OpenAIConfiguration


# TODO: Change client to use tools instead of processing the raw files
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

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(1))
    def __call__(
        self,
        prompt: Union[str, List[Dict[str, str]]],
    ) -> str:
        """Calls the OpenAI API.

        Args:
            prompt: The prompt to use. Either a list of messages or a string.
        """
        if isinstance(prompt, str):
            prompt = [{"role": "system", "content": prompt}]
        answers = asyncio.run(
            self.__openai_client.chat.completions.create(
                model=self.config.model,
                messages=prompt,  # type: ignore
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        )
        if (
            not answers.choices
            or not answers.choices[0].message
            or not answers.choices[0].message.content
        ):
            raise ValueError("OpenAI did not return any completions.")
        return answers.choices[0].message.content

    async def call_async(
        self,
        prompt: Union[str, List[Dict[str, str]]],
    ) -> str:
        """Calls the OpenAI API asynchronously.

        Args:
            prompt: The prompt to use. Either a list of messages or a string.
        """
        if isinstance(prompt, str):
            prompt = [{"role": "system", "content": prompt}]
        answers = await self.__openai_client.chat.completions.create(
            model=self.config.model,
            messages=prompt,  # type: ignore
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        if (
            not answers.choices
            or not answers.choices[0].message
            or not answers.choices[0].message.content
        ):
            raise ValueError("OpenAI did not return any completions.")
        return answers.choices[0].message.content

    @staticmethod
    def __get_openai_client(openai_config: OpenAIConfiguration) -> AsyncOpenAI:
        if openai_config.api_type == "azure":
            if openai_config.api_base is None:
                raise ValueError("Azure-OpenAI base url not found in configuration.")
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
