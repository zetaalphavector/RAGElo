import asyncio
from typing import Dict, List, Union

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.types import LLMProviderTypes
from ragelo.types.configurations import OllamaConfiguration


# TODO: Change client to use tools instead of processing the raw files
@LLMProviderFactory.register(LLMProviderTypes.OLLAMA)
class OllamaProvider(BaseLLMProvider):
    """A Wrapper over the Ollama client."""

    config: OllamaConfiguration
    api_key_env_var: str = "OPENAI_API_KEY"

    def __init__(
        self,
        config: OllamaConfiguration,
    ):
        super().__init__(config)
        self.__ollama_client = self.__get_ollama_client(config)

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(1))
    def __call__(
        self,
        prompt: Union[str, List[Dict[str, str]]],
    ) -> str:
        """Calls the OpenAI API.

        Args:
            prompt: The prompt to use. Either a list of messages or a string.
            temperature : The temperature to use. Defaults to 0.1.
            max_tokens: The maximum number of tokens to retrieve. Defaults 512.
        """
        if isinstance(prompt, str):
            prompt = [{"role": "system", "content": prompt}]
        answers = asyncio.run(
            self.__ollama_client.chat.completions.create(
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
            raise ValueError("Ollama did not return any completions.")
        return answers.choices[0].message.content

    async def call_async(
        self,
        prompt: Union[str, List[Dict[str, str]]],
    ) -> str:
        """Calls the Ollama Local API asynchronously.

        Args:
            prompt: The prompt to use. Either a list of messages or a string.
            temperature : The temperature to use. Defaults to 0.1.
            max_tokens: The maximum number of tokens to retrieve. Defaults 512.
        """
        if isinstance(prompt, str):
            prompt = [{"role": "system", "content": prompt}]
        answers = await self.__ollama_client.chat.completions.create(
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
    def __get_ollama_client(ollama_config: OllamaConfiguration) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=ollama_config.api_base,
            api_key=ollama_config.api_key,
        )
