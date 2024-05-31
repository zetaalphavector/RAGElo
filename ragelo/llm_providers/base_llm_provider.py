"""A LLM provider is a class that can be called with a string and returns with another string as an answer from an LLM model."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Union, get_type_hints

from ragelo.types import LLMProviderConfig, LLMProviderTypes


def set_credentials_from_file(credentials_file: str, split_char: str = "="):
    """Read credentials from a file and add them to the environment"""
    logging.info(f"Loading credentials from {credentials_file}")
    if not os.path.isfile(credentials_file):
        raise FileNotFoundError(f"Credentials file {credentials_file} not found")
    with open(credentials_file) as f:
        for line in f:
            key, value = line.strip().split(split_char, 1)
            logging.debug(f"Setting {key} from file")
            os.environ[key] = value
            os.environ[key] = value


class BaseLLMProvider(ABC):
    config: LLMProviderConfig
    api_key_env_var: str = "API_KEY"

    def __init__(self, config: LLMProviderConfig):
        self.config = config

    @abstractmethod
    def __call__(self, prompt: Union[str, List[Dict[str, str]]]) -> str:
        """Submits a single query-document pair to the LLM and returns the answer."""
        raise NotImplementedError

    @abstractmethod
    async def call_async(
        self,
        prompt: Union[str, List[Dict[str, str]]],
    ) -> str:
        """Submits a single query-document pair to the LLM and returns the answer."""
        raise NotImplementedError

    @classmethod
    def from_config(
        cls,
        config: LLMProviderConfig,
    ) -> "BaseLLMProvider":
        """Inits the LLM provider from a credentials file."""
        return cls(config)

    @classmethod
    def get_config_class(cls) -> Type[LLMProviderConfig]:
        return get_type_hints(cls)["config"]


class LLMProviderFactory:
    registry: Dict[LLMProviderTypes, Type[BaseLLMProvider]] = {}

    @classmethod
    def register(cls, name: LLMProviderTypes):
        """Registers a new LLM provider"""

        def inner_wrapper(
            wrapped_class: Type[BaseLLMProvider],
        ) -> Type[BaseLLMProvider]:
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        name: LLMProviderTypes,
        config: Optional[LLMProviderConfig] = None,
        credentials_file: Optional[str] = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """Creates a new LLM provider"""
        if name not in cls.registry:
            raise ValueError(f"LLM provider {name} not found")
        if credentials_file and os.path.isfile(credentials_file):
            set_credentials_from_file(credentials_file)
        if config is None:
            class_ = cls.registry[name]
            type_config = class_.get_config_class()
            valid_keys = [field for field in type_config.get_model_fields()]
            if "api_key" not in kwargs:
                api_key = os.environ.get(class_.api_key_env_var)
                if not api_key:
                    raise ValueError(
                        f"API key not found in environment variable {class_.api_key_env_var}"
                    )
                kwargs["api_key"] = api_key
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            config = type_config(**valid_args)
        return cls.registry[name].from_config(config)


def get_llm_provider(
    name: Union[LLMProviderTypes, str],
    config: Optional[LLMProviderConfig] = None,
    credentials_file: Optional[str] = None,
    **kwargs,
) -> BaseLLMProvider:
    """Creates a new LLM provider"""
    if isinstance(name, str):
        name = LLMProviderTypes(name)
    return LLMProviderFactory.create(name, config, credentials_file, **kwargs)
