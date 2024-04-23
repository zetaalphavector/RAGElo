import asyncio

from aiohttp import ClientError, ClientSession
from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.logger import logger
from ragelo.types import LLMProviderTypes
from ragelo.types.configurations import OpenAIConfiguration


# TODO: Change client to use json_mode
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
        prompt: str | list[dict[str, str]],
    ) -> str:
        """Calls the OpenAI API.

        Args:
            prompt: The prompt to use. Either a list of messages or a string.
            temperature : The temperature to use. Defaults to 0.1.
            max_tokens: The maximum number of tokens to retrieve. Defaults 512.
        """
        if isinstance(prompt, str):
            prompt = [{"role": "system", "content": prompt}]
        answers = self.__openai_client.chat.completions.create(
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

    async def call_async(
        self,
        prompt: str | list[dict[str, str]],
        session: ClientSession,
    ) -> str:
        """Calls the OpenAI API asynchronously.

        Args:
            prompt: The prompt to use. Either a list of messages or a string.
            temperature : The temperature to use. Defaults to 0.1.
            max_tokens: The maximum number of tokens to retrieve. Defaults 512.
        """
        if isinstance(prompt, str):
            prompt = [{"role": "system", "content": prompt}]
        retries = 0
        payload = {
            "model": self.config.model,
            "messages": prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        url = f"{self.__openai_client.base_url}chat/completions"
        headers = {
            "Authorization": f"Bearer {self.__openai_client.api_key}",
            "Content-Type": "application/json",
        }
        if self.__openai_client.organization:
            headers["OpenAI-Organization"] = self.__openai_client.organization
        while retries < self.config.max_retries:
            try:
                async with session.post(url, json=payload, headers=headers) as response:
                    # Check for response status if necessary, e.g., raise exception for 4xx/5xx
                    response.raise_for_status()
                    json_response = await response.json()  # Parsing response as JSON
                    return json_response["choices"][0]["message"]["content"]
            except (ClientError, asyncio.TimeoutError) as e:
                logger.info(
                    f"Request failed: {e}, retrying... ({retries+1}/{self.config.max_retries})"
                )
                retries += 1
                await asyncio.sleep(self.config.sleep_time)
            except Exception as e:
                logger.error(f"An unexpected error occurred: {e}")
                raise
        raise ValueError("OpenAI did not return any completions.")

    @staticmethod
    def __get_openai_client(openai_config: OpenAIConfiguration) -> OpenAI:
        if openai_config.api_type == "azure":
            if openai_config.api_base is None:
                raise ValueError("OpenAI base url not found in configuration")
            return AzureOpenAI(
                azure_endpoint=openai_config.api_base,
                api_key=openai_config.api_key,
                api_version=openai_config.api_version,
            )
        elif openai_config.api_type == "open_ai" or openai_config.api_type is None:
            return OpenAI(
                base_url=openai_config.api_base,
                api_key=openai_config.api_key,
                organization=openai_config.org,
            )
        else:
            raise Exception(f"Unknown OpenAI api type: {openai_config.api_type}")
