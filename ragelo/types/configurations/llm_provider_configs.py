from typing import Optional

from pydantic import BaseModel


class LLMProviderConfig(BaseModel):
    api_key: str


class OpenAIConfiguration(LLMProviderConfig):
    org: Optional[str] = None
    api_type: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    model: str = "gpt-4-turbo"
