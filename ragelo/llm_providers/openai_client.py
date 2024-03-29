from openai import AzureOpenAI, OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.types import LLMProviderTypes
from ragelo.types.configurations import OpenAIConfiguration


@LLMProviderFactory.register(LLMProviderTypes.OPENAI)
class OpenAIProvider(BaseLLMProvider):
    """A Wrapper over the OpenAI client."""

    config: OpenAIConfiguration

    def __init__(
        self,
        config: OpenAIConfiguration,
    ):
        super().__init__(config)
        self.__openai_client = self.get_openai_client(config)

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(1))
    def __call__(
        self,
        prompt: str | list[dict[str, str]],
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
            model=self.config.model_name,
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

    @staticmethod
    def get_openai_client(openai_config: OpenAIConfiguration) -> OpenAI:
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

    def set_openai_client(self, openai_client: OpenAI):
        self.__openai_client = openai_client
