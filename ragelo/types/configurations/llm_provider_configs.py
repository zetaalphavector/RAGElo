from typing import Literal

from pydantic import BaseModel, Field, SecretStr


class LLMProviderConfig(BaseModel):
    temperature: float | None = None
    max_tokens: int = 4096
    json_mode: bool = False
    reasoning_effort: Literal["low", "medium", "high"] | None = None


class OpenAIConfiguration(LLMProviderConfig):
    api_key: SecretStr
    org: str | None = None
    api_type: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    model: str = "gpt-5.6-luna"


class VercelConfiguration(OpenAIConfiguration):
    api_base: str | None = "https://ai-gateway.vercel.sh/v1"
    model: str


class JevConfiguration(LLMProviderConfig):
    batch_size: int = Field(
        default=10,
        ge=1,
        description="How many prompts with one batch key, such as the documents of one query, are asked in one request.",
    )
    batch_wait: float = Field(
        default=0.02, ge=0, description="Seconds a prompt waits for others to share its request."
    )


class VercelJevConfiguration(JevConfiguration):
    api_key: SecretStr
    api_base: str = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
    model: str = "typesafe-ai/jev"
    timeout: float = 60.0
    max_retries: int = 2


class TypeSafeConfiguration(JevConfiguration):
    api_key: SecretStr
    api_base: str | None = None
    model: str = "jev-latest"
    timeout: float = 60.0
    max_retries: int = 2


class OllamaConfiguration(LLMProviderConfig):
    api_base: str | None = "http://localhost:11434/v1/"
    model: str
    seed: int | None = 42


class InstructorConfiguration(LLMProviderConfig):
    model: str
    api_key: SecretStr | None = None
    max_retries: int = 3
    use_cache: bool = True
    cache_size: int = 1000
    model_kwargs: dict[str, str] = {}
