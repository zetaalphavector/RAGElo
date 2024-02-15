import os
from typing import Dict, List, Optional

from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import (
    BaseLLMProvider,
    set_credentials_from_file,
)
from ragelo.types import OpenAIConfiguration


class OpenAIProvider(BaseLLMProvider):
    """A Wrapper over the OpenAI client."""

    def __init__(self, openai_client: OpenAI | AzureOpenAI, model: str):
        self.__model = model
        self.__openai_client = openai_client

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(1))
    def __call__(
        self,
        prompt: str | List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
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
            model=self.__model,
            messages=prompt,  # type: ignore
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if (
            not answers.choices
            or not answers.choices[0].message
            or not answers.choices[0].message.content
        ):
            raise ValueError("OpenAI did not return any completions.")
        return answers.choices[0].message.content

    @classmethod
    def from_configuration(cls, openai_config: OpenAIConfiguration):
        """Inits the OpenAI wrapper from a configuration object."""
        openai_client = cls.get_openai_client(openai_config)
        return cls(openai_client, model=openai_config.model_name)

    @staticmethod
    def get_openai_client(openai_config: OpenAIConfiguration) -> OpenAI:
        if openai_config.openai_api_type == "azure":
            if openai_config.openai_api_base is None:
                raise ValueError("OpenAI base url not found in configuration")
            return AzureOpenAI(
                azure_endpoint=openai_config.openai_api_base,
                api_key=openai_config.api_key,
                api_version=openai_config.openai_api_version,
            )
        elif (
            openai_config.openai_api_type == "open_ai"
            or openai_config.openai_api_type is None
        ):
            return OpenAI(
                base_url=openai_config.openai_api_base,
                api_key=openai_config.api_key,
                organization=openai_config.openai_org,
            )
        else:
            raise Exception(f"Unknown OpenAI api type: {openai_config.openai_api_type}")

    @staticmethod
    def get_openai_config(
        credential_file: Optional[str], model_name: str
    ) -> OpenAIConfiguration:
        """Get the OpenAI configuration."""
        if credential_file:
            set_credentials_from_file(credential_file)
        return OpenAIConfiguration(
            api_key=os.getenv("OPENAI_API_KEY", "fake key"),
            openai_org=os.getenv("OPENAI_ORG"),
            openai_api_type=os.getenv("OPENAI_API_TYPE"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            model_name=model_name,
        )
