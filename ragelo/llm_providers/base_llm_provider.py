from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Type, get_type_hints

from pydantic import BaseModel as PydanticBaseModel

from ragelo.types.configurations import LLMProviderConfig
from ragelo.types.formats import AnswerFormat, LLMResponseType
from ragelo.types.pydantic_models import _PYDANTIC_MAJOR_VERSION
from ragelo.types.types import LLMProviderTypes
from ragelo.utils import call_async_fn


class BaseLLMProvider(ABC):
    config: LLMProviderConfig
    api_key_env_var: str = "API_KEY"

    def __init__(self, config: LLMProviderConfig):
        self.config = config

    def __call__(
        self,
        prompt: str | list[dict[str, str]],
        answer_format: AnswerFormat = AnswerFormat.TEXT,
        response_schema: Type[PydanticBaseModel] | dict[str, Any] | None = None,
    ) -> LLMResponseType:
        """Submits a single query-document pair to the LLM and returns the answer."""
        return call_async_fn(self.call_async, prompt, answer_format, response_schema)

    @abstractmethod
    async def call_async(
        self,
        prompt: str | list[dict[str, str]],
        answer_format: AnswerFormat = AnswerFormat.TEXT,
        response_schema: Type[PydanticBaseModel] | dict[str, Any] | None = None,
    ) -> LLMResponseType:
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
    registry: dict[LLMProviderTypes, Type[BaseLLMProvider]] = {}

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
        config: LLMProviderConfig | None = None,
        **kwargs,
    ) -> BaseLLMProvider:
        """Creates a new LLM provider"""
        if name not in cls.registry:
            raise ValueError(f"LLM provider {name} not found")
        if config is None:
            class_ = cls.registry[name]
            type_config = class_.get_config_class()
            valid_keys = [field for field in type_config.get_model_fields()]
            if "api_key" not in kwargs:
                api_key = os.environ.get(class_.api_key_env_var)
                if not api_key:
                    # Check if the key is actually required
                    api_key_field = type_config.get_model_fields()["api_key"]
                    if _PYDANTIC_MAJOR_VERSION == 2:
                        is_required = api_key_field.is_required()
                    else:
                        is_required = api_key_field.required  # type: ignore
                    if is_required:
                        raise ValueError(f"API key not found in environment variable {class_.api_key_env_var}")
                    else:
                        api_key = api_key_field.default
                kwargs["api_key"] = api_key
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            config = type_config(**valid_args)
        return cls.registry[name].from_config(config)


def get_llm_provider(
    name: LLMProviderTypes | str,
    config: LLMProviderConfig | None = None,
    **kwargs,
) -> BaseLLMProvider:
    """Creates a new LLM provider"""
    if isinstance(name, str):
        name = LLMProviderTypes(name)
    return LLMProviderFactory.create(name, config, **kwargs)
