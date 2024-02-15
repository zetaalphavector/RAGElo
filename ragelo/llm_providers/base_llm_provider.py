"""A LLM provider is a class that can be called with a string and returns with another string as an answer from an LLM model."""

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List


class BaseLLMProvider(ABC):
    @abstractmethod
    def __call__(self, prompt: str | List[Dict[str, str]]) -> str:
        """Submits a single query-document pair to the LLM and returns the answer."""
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
