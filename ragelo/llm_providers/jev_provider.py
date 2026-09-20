from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import LLMInputPrompt, LLMResponseType
from ragelo.types.configurations import JevConfiguration
from ragelo.types.formats import JevResponse, LLMUsage

T_Schema = TypeVar("T_Schema", bound=BaseModel)
JevState = str | dict[str, str]
JevQuestions = dict[str, dict[str, Any]]


@dataclass(slots=True)
class _Batch:
    """Prompts with one `batch_key` that arrived together, each with the future its caller awaits."""

    prompts: list[LLMInputPrompt] = field(default_factory=list)
    futures: list[asyncio.Future[LLMResponseType[JevResponse]]] = field(default_factory=list)
    timer: asyncio.TimerHandle | None = None


class JevProvider(BaseLLMProvider):
    """TypeSafe's Jev answers typed questions about a state with probabilities and generates no text, so only
    the jev evaluators can use it. The subclasses differ in how they reach it, and all return a `JevResponse`.

    Prompts that share a `batch_key` and arrive within `config.batch_wait` seconds are asked in one request, up
    to `config.batch_size` of them. On LLMJudge that halves the input tokens, and documents of one query are
    judged better together than alone, while documents of different queries are not (benchmarks/README.md).
    A batch only forms from calls that are in flight together, so it needs `n_processes` of that size.
    """

    config: JevConfiguration

    def __init__(self, config: JevConfiguration) -> None:
        super().__init__(config)
        self.__waiting: dict[str, _Batch] = {}

    @abstractmethod
    async def _request(self, state: JevState, questions: JevQuestions) -> LLMResponseType[JevResponse]:
        raise NotImplementedError

    async def call_async(self, input: LLMInputPrompt, response_schema: type[T_Schema]) -> LLMResponseType[T_Schema]:
        """Returns a JevResponse as the parsed answer. The jev evaluators turn it into `response_schema`."""
        if input.batch_key is None or self.config.batch_size < 2:
            return await self._request(self._state(input), self._questions(input))  # type: ignore[return-value]
        loop = asyncio.get_running_loop()
        key = input.batch_key
        batch = self.__waiting.setdefault(key, _Batch())
        future: asyncio.Future[LLMResponseType[JevResponse]] = loop.create_future()
        batch.prompts.append(input)
        batch.futures.append(future)
        if batch.timer is None:
            batch.timer = loop.call_later(self.config.batch_wait, self.__send, key)
        if len(batch.prompts) >= self.config.batch_size:
            self.__send(key)
        return await future  # type: ignore[return-value]

    def __send(self, key: str) -> None:
        batch = self.__waiting.pop(key, None)
        if batch is None:
            return
        if batch.timer is not None:
            batch.timer.cancel()
        asyncio.ensure_future(self.__answer(batch))

    async def __answer(self, batch: _Batch) -> None:
        """A merged request that fails, say for being too long, is asked again one prompt at a time."""
        responses: Sequence[LLMResponseType[JevResponse] | BaseException]
        try:
            responses = await self.__ask_together(batch.prompts)
        except Exception:  # noqa: BLE001 - every prompt gets its own outcome below
            responses = await asyncio.gather(
                *(self._request(self._state(p), self._questions(p)) for p in batch.prompts), return_exceptions=True
            )
        for future, response in zip(batch.futures, responses):
            if isinstance(response, BaseException):
                future.set_exception(response)
            else:
                future.set_result(response)

    async def __ask_together(self, prompts: list[LLMInputPrompt]) -> list[LLMResponseType[JevResponse]]:
        if len(prompts) == 1:
            return [await self._request(self._state(prompts[0]), self._questions(prompts[0]))]
        state = {f"item_{i}": self._state(prompt) for i, prompt in enumerate(prompts)}
        questions = {
            f"item_{i}__{name}": question
            | {"instructions": f"Consider only `item_{i}`. {question['instructions'] or ''}"}
            for i, prompt in enumerate(prompts)
            for name, question in self._questions(prompt).items()
        }
        merged = await self._request(state, questions)
        usages = self.__shares(merged.usage, len(prompts))
        responses: list[LLMResponseType[JevResponse]] = []
        for i, prompt in enumerate(prompts):
            answers = {name: merged.parsed_answer.answers[f"item_{i}__{name}"] for name in self._questions(prompt)}
            parsed = JevResponse(answers=answers)
            responses.append(
                LLMResponseType(raw_answer=parsed.model_dump_json(), parsed_answer=parsed, usage=usages[i])
            )
        return responses

    @staticmethod
    def __shares(usage: LLMUsage | None, n: int) -> list[LLMUsage | None]:
        """Equal shares of a request's tokens, with the remainder on the first so the shares add up to it."""
        if usage is None:
            return [None] * n
        shares: list[LLMUsage | None] = []
        for i in range(n):
            extra = 1 if i == 0 else 0
            shares.append(
                LLMUsage(
                    input_tokens=usage.input_tokens // n + extra * (usage.input_tokens % n),
                    output_tokens=usage.output_tokens // n + extra * (usage.output_tokens % n),
                    cached_tokens=usage.cached_tokens // n + extra * (usage.cached_tokens % n),
                )
            )
        return shares

    @staticmethod
    def _state(input: LLMInputPrompt) -> str:
        if input.user_message is None:
            raise ValueError("Jev judges a state, which the jev evaluators pass as the user message.")
        return input.user_message

    @staticmethod
    def _questions(input: LLMInputPrompt) -> JevQuestions:
        """The system prompt is the instruction of every question that does not bring its own."""
        if not input.questions:
            raise ValueError("Jev needs typed questions. Use it with one of the jev evaluators.")
        return {name: {"instructions": input.system_prompt, **question} for name, question in input.questions.items()}
