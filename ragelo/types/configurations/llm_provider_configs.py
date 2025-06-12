from __future__ import annotations

from pydantic import BaseModel


class LLMProviderConfig(BaseModel):
    api_key: str | None
    temperature: float = 0.1
    max_tokens: int = 2048
    seed: int | None = 42


class OpenAIConfiguration(LLMProviderConfig):
    organization: str | None = None
    api_type: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    model: str = "gpt-4o-mini"
    azure_endpoint: str | None = None
    azure_deployment: str | None = None
    project: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    max_retries: int | None = None
    default_headers: dict[str, str] | None = None
    default_query: dict[str, str] | None = None


class OllamaConfiguration(LLMProviderConfig):
    organization: str | None = None
    api_base: str | None = "http://localhost:11434/v1/"
    api_version: str | None = None
    api_key: str | None = "NoKeyNeeded"
    model: str = "gemma2:27b"
