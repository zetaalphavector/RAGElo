import os
from typing import Dict, List, Optional, Union

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ragelo.logger import logger


class OpenAiClient:
    """A Wrapper over the OpenAI client."""

    def __init__(
        self,
        model: Optional[str] = "gpt-3.5-turbo",
        request_type: Optional[str] = "chat",
    ):
        """Initialize the OpenAI client.

        Args:
            key: The API key
            model: The model to use. Defaults to "gpt-3.5-turbo".
            request_type: The type of request. Defaults to "chat".
            type: The type of API to use. Defaults to "open_ai".
            base_url: The base URL to use. Defaults to "https: //api.openai.com/v1".
            version: The version of the API to use. Defaults to None.

        """
        self.openai_type = os.environ.get("OPENAI_TYPE") or "open_ai"
        if self.openai_type == "azure":
            service = os.environ.get("OPENAI_SERVICE_GPT4") or os.environ.get(
                "OPENAI_SERVICE"
            )
            base_url = f"https://{service}.openai.azure.com/"
        else:
            base_url = openai.api_base

        key = os.environ.get("OPENAI_API_KEY_GPT4") or os.environ.get("OPENAI_API_KEY")
        model = (
            model
            or os.environ.get("OPENAI_GPT4_DEPLOYMENT")
            or os.environ.get("OPENAI_GPT_DEPLOYMENT")
        )
        if not key:
            raise Exception("No API key found")
        if not model:
            raise Exception("No model found")

        openai.api_type = self.openai_type
        openai.api_version = os.environ.get("OPENAI_VERSION") or openai.api_version
        openai.api_base = base_url or openai.api_base
        openai.api_key = key
        self.model = model
        self.completion_type = request_type
        self.__print_credentials(key, self.model, openai.api_base, openai.api_type)

    @retry(wait=wait_random_exponential(min=1, max=120), stop=stop_after_attempt(1))
    def __call__(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        """Calls the OpenAI API.

        Args:
            prompt: The prompt to use.
            temperature : The temperature to use. Defaults to 0.7.
            max_tokens: The maximum number of tokens to use. Defaults to 256.
        """
        if isinstance(prompt, str):
            prompt = [{"role": "system", "content": prompt}]

        if self.openai_type == "azure":
            if self.completion_type == "completion":
                answers = openai.Completion.create(
                    engine=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif self.completion_type == "chat":
                answers = openai.ChatCompletion.create(
                    engine=self.model,
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise Exception(f"Unknown completion type: {self.completion_type}")
        elif self.openai_type == "open_ai":
            if self.completion_type == "completion":
                answers = openai.Completion.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            elif self.completion_type == "chat":
                answers = openai.ChatCompletion.create(
                    model=self.model,
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise Exception(f"Unknown completion type: {self.completion_type}")
        else:
            raise Exception(f"Unknown type: {self.openai_type}")
        full_text = answers.choices[0].message.content  # type: ignore
        return full_text

    def __print_credentials(
        self,
        key: str,
        model: str,
        base_url: Optional[str] = None,
        api_type: Optional[str] = None,
    ):
        """Prints credentials to scren."""
        logger.info("Starting OPENAI client with the following settings:")
        logger.info(f":lock: [red]API KEY [/red]  : {key[:3]}...{key[-3:]}")
        logger.info(
            f":brain: [bold]Model[/bold]     : [not bold white]{model}[/not bold white]"
        )
        logger.info(f":globe_with_meridians: [bold]API Type[/bold]  : {api_type}")
        logger.info(f":link: [bold]API URL [/bold]   : {base_url}")


def set_credentials_from_file(credentials_file: str):
    """Read credentials from a file and add them to the environment."""
    logger.info(f"Loading credentials from {credentials_file}")
    if not os.path.isfile(credentials_file):
        raise FileNotFoundError(f"Credentials file {credentials_file} not found")
    with open(credentials_file) as f:
        for line in f:
            key, value = line.strip().split("=")
            logger.debug(f"Setting {key} from file")
            os.environ[key] = value
            os.environ[key] = value
