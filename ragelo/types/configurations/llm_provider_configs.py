from typing import Literal

from pydantic import BaseModel, SecretStr


class LLMProviderConfig(BaseModel):
    temperature: float | None = None
    max_tokens: int = 4096
    seed: int | None = 42
    json_mode: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = None


class OpenAIConfiguration(LLMProviderConfig):
    api_key: SecretStr
    org: str | None = None
    api_type: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    model: str = "gpt-4.1-mini"


class OllamaConfiguration(LLMProviderConfig):
    api_base: str | None = "http://localhost:11434/v1/"
    model: str


class InstructorConfiguration(LLMProviderConfig):
    model: str
    api_key: SecretStr | None = None
    max_retries: int = 3
    use_cache: bool = True
    cache_size: int = 1000
    model_kwargs: dict[str, str] = {}
