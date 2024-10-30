from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel as PydanticBaseModel

from ragelo.types.configurations.base_configs import BaseModel


class LLMProviderConfig(BaseModel):
    api_key: str | None
    max_retries: int = 3
    sleep_time: int = 2
    temperature: float = 0.1
    max_tokens: int = 2048


class OpenAIConfiguration(LLMProviderConfig):
    org: str | None = None
    api_type: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    model: str = "gpt-4o-mini"
    response_schema: Type[PydanticBaseModel] | dict[str, Any] | None = None


class OllamaConfiguration(LLMProviderConfig):
    org: str | None = None
    api_base: str | None = "http://localhost:11434/v1/"
    api_version: str | None = None
    api_key: str | None = "NoKeyNeeded"
    model: str = "gemma2:27b"
