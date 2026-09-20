from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from pydantic import BaseModel

from ragelo.llm_providers.base_llm_provider import LLMProviderFactory
from ragelo.llm_providers.jev_provider import JevProvider, JevQuestions, JevState
from ragelo.types import LLMResponseType
from ragelo.types.configurations import VercelJevConfiguration
from ragelo.types.formats import JevAnswer, JevResponse, LLMUsage
from ragelo.types.types import LLMProviderTypes

T_Schema = TypeVar("T_Schema", bound=BaseModel)
RETRIED_STATUSES = {
    httpx.codes.TOO_MANY_REQUESTS,
    httpx.codes.BAD_GATEWAY,
    httpx.codes.SERVICE_UNAVAILABLE,
    httpx.codes.GATEWAY_TIMEOUT,
}


@LLMProviderFactory.register(LLMProviderTypes.VERCEL_JEV)
class VercelJevProvider(JevProvider):
    """Jev through the Vercel AI Gateway's evaluation endpoint, which Vercel documents for its AI SDK only."""

    config: VercelJevConfiguration
    api_key_env_var: str = "AI_GATEWAY_API_KEY"

    def __init__(
        self,
        config: VercelJevConfiguration,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__(config)
        self.__client = client or httpx.AsyncClient(timeout=config.timeout)
        self.__sleep = sleep

    async def _request(self, state: JevState, questions: JevQuestions) -> LLMResponseType[JevResponse]:
        response = await self.__post({"state": state, "questions": questions})
        if response.is_error:
            raise ValueError(f"Jev request failed: {response.status_code} {response.text}")
        try:
            body = response.json()
            confidence = body.get("providerMetadata", {}).get("typesafe", {}).get("confidence", {})
            parsed = JevResponse(
                answers={
                    name: JevAnswer(**{"confidence": confidence.get(name)} | answer)
                    for name, answer in body["answers"].items()
                }
            )
            usage = None
            if "usage" in body:
                usage = LLMUsage(
                    input_tokens=body["usage"]["inputTokens"], output_tokens=body["usage"]["outputTokens"]
                )
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise ValueError(f"Jev answered in an unexpected format ({type(e).__name__}: {e}): {response.text}") from e
        return LLMResponseType(raw_answer=response.text, parsed_answer=parsed, usage=usage)

    async def __post(self, body: dict[str, object]) -> httpx.Response:
        """Under load the gateway answers 503 or stops answering. Those, the other statuses worth a second try
        and transport errors are retried with the TypeSafe SDK's backoff. The last attempt's response is
        returned and its transport error raised."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
            "ai-model-id": self.config.model,
            "ai-evaluation-model-specification-version": "4",
            "ai-gateway-protocol-version": "0.0.1",
        }
        for attempt in range(self.config.max_retries):
            try:
                response = await self.__client.post(self.config.api_base, headers=headers, json=body)
            except httpx.TransportError:
                response = None
            if response is not None and response.status_code not in RETRIED_STATUSES:
                return response
            await self.__sleep(min(0.5 * 2**attempt, 5.0))
        return await self.__client.post(self.config.api_base, headers=headers, json=body)
