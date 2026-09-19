from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from pydantic import BaseModel

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, LLMProviderFactory
from ragelo.types import LLMInputPrompt, LLMResponseType
from ragelo.types.configurations import VercelJevConfiguration
from ragelo.types.formats import JevAnswer, JevResponse, LLMUsage
from ragelo.types.types import LLMProviderTypes

T_Schema = TypeVar("T_Schema", bound=BaseModel)


@LLMProviderFactory.register(LLMProviderTypes.VERCEL_JEV)
class VercelJevProvider(BaseLLMProvider):
    """TypeSafe's Jev through the Vercel AI Gateway. Jev answers typed questions about a state
    with probabilities and generates no text, so only the jev evaluators can use it.
    """

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

    async def call_async(self, input: LLMInputPrompt, response_schema: type[T_Schema]) -> LLMResponseType[T_Schema]:
        """Returns a JevResponse as the parsed answer. The jev evaluators turn it into `response_schema`."""
        if not input.questions:
            raise ValueError("Jev needs typed questions. Use it with one of the jev evaluators.")
        questions = {
            name: {"instructions": input.system_prompt, **question} for name, question in input.questions.items()
        }
        response = await self.__post({"state": input.user_message, "questions": questions})
        if response.is_error:
            raise ValueError(f"Jev request failed: {response.status_code} {response.text}")
        body = response.json()
        confidence = body.get("providerMetadata", {}).get("typesafe", {}).get("confidence", {})
        parsed = JevResponse(
            answers={
                name: JevAnswer(**answer, confidence=confidence.get(name)) for name, answer in body["answers"].items()
            }
        )
        usage = None
        if "usage" in body:
            usage = LLMUsage(input_tokens=body["usage"]["inputTokens"], output_tokens=body["usage"]["outputTokens"])
        return LLMResponseType(raw_answer=response.text, parsed_answer=parsed, usage=usage)  # type: ignore[arg-type]

    async def __post(self, body: dict[str, object]) -> httpx.Response:
        """The gateway answers 503 under load, so those are retried with the TypeSafe SDK's backoff."""
        for attempt in range(self.config.max_retries + 1):
            response = await self.__client.post(
                self.config.api_base,
                headers={
                    "Authorization": f"Bearer {self.config.api_key.get_secret_value()}",
                    "ai-model-id": self.config.model,
                    "ai-evaluation-model-specification-version": "4",
                    "ai-gateway-protocol-version": "0.0.1",
                },
                json=body,
            )
            if response.status_code != httpx.codes.SERVICE_UNAVAILABLE or attempt == self.config.max_retries:
                break
            await self.__sleep(min(0.5 * 2**attempt, 5.0))
        return response
