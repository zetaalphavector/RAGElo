from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, SecretStr


class LLMProviderConfig(BaseModel):
    api_key: SecretStr | None
    temperature: float | None = 0.1
    max_tokens: int = 2048
    seed: int | None = 42
    json_mode: bool = False
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None


class OpenAIConfiguration(LLMProviderConfig):
    org: str | None = None
    api_type: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    model: str = "gpt-4.1-mini"


class OllamaConfiguration(LLMProviderConfig):
    org: str | None = None
    api_base: str | None = "http://localhost:11434/v1/"
    api_version: str | None = None
    api_key: SecretStr | None = SecretStr("NoKeyNeeded")
    model: str = "gemma2:27b"
