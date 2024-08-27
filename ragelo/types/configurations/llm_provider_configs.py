from typing import Optional

from ragelo.types.configurations.base_configs import BaseModel


class LLMProviderConfig(BaseModel):
    api_key: Optional[str]
    max_retries: int = 3
    sleep_time: int = 2
    temperature: float = 0.1
    max_tokens: int = 512


class OpenAIConfiguration(LLMProviderConfig):
    org: Optional[str] = None
    api_type: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    model: str = "gpt-4o-mini"


class OllamaConfiguration(LLMProviderConfig):
    org: Optional[str] = None
    api_base: Optional[str] = "http://localhost:11434/v1/"
    api_version: Optional[str] = None
    api_key: Optional[str] = "NoKeyNeeded"
    model: str = "gemma2:27b"
