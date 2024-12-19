from __future__ import annotations

from ragelo.types.configurations.base_configs import BaseModel


class LLMProviderConfig(BaseModel):
    api_key: str | None
    temperature: float = 0.1
    max_tokens: int = 2048
    seed: int | None = 42


class OpenAIConfiguration(LLMProviderConfig):
    org: str | None = None
    api_type: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    model: str = "gpt-4o-mini"


class OllamaConfiguration(LLMProviderConfig):
    org: str | None = None
    api_base: str | None = "http://localhost:11434/v1/"
    api_version: str | None = None
    api_key: str | None = "NoKeyNeeded"
    model: str = "gemma2:27b"
