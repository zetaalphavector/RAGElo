"""A LLM provider is a class that can be called with a string and returns with another string as an answer from an LLM model."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ragelo.types import LLMProviderConfiguration


class BaseLLMProvider(ABC):
    @abstractmethod
    def __call__(self, prompt: str | List[Dict[str, str]]) -> str:
        """Submits a single query-document pair to the LLM and returns the answer."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: LLMProviderConfiguration):
        """Inits the LLM provider from a configuration object."""
        raise NotImplementedError

    @classmethod
    def from_credentials_file(
        cls, credentials_file: Optional[str], model_name: str
    ) -> "BaseLLMProvider":
        """Inits the LLM provider from a credentials file."""
        raise NotImplementedError


def set_credentials_from_file(credentials_file: str):
    """Read credentials from a file and add them to the environment"""
    logging.info(f"Loading credentials from {credentials_file}")
    if not os.path.isfile(credentials_file):
        raise FileNotFoundError(f"Credentials file {credentials_file} not found")
    with open(credentials_file) as f:
        for line in f:
            key, value = line.strip().split("=")
            logging.debug(f"Setting {key} from file")
            os.environ[key] = value
            os.environ[key] = value


class LLMProviderFactory:
    registry: Dict[str, BaseLLMProvider] = {}

    @classmethod
    def register(cls, name: str):
        """Registers a new LLM provider"""

        def decorator(llm_provider: BaseLLMProvider):
            cls.registry[name] = llm_provider
            return llm_provider

        return decorator

    @classmethod
    def create(cls, name: str, config: LLMProviderConfiguration) -> BaseLLMProvider:
        """Creates a new LLM provider"""
        if name not in cls.registry:
            raise ValueError(f"LLM provider {name} not found")
        return cls.registry[name].from_config(config)

    @classmethod
    def create_from_credentials_file(
        cls,
        name: str,
        credentials_file: Optional[str],
        model_name: str,
    ) -> BaseLLMProvider:
        """Creates a new LLM provider"""
        if name not in cls.registry:
            raise ValueError(f"LLM provider {name} not found")
        return cls.registry[name].from_credentials_file(credentials_file, model_name)
