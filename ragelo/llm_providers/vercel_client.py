from __future__ import annotations

from ragelo.llm_providers.base_llm_provider import LLMProviderFactory
from ragelo.llm_providers.openai_client import OpenAIProvider
from ragelo.types.configurations import VercelConfiguration
from ragelo.types.types import LLMProviderTypes


@LLMProviderFactory.register(LLMProviderTypes.VERCEL)
class VercelProvider(OpenAIProvider):
    """The OpenAI Responses client pointed at the Vercel AI Gateway."""

    config: VercelConfiguration
    api_key_env_var: str = "AI_GATEWAY_API_KEY"
