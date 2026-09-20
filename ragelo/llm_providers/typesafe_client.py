from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel

from ragelo.llm_providers.base_llm_provider import LLMProviderFactory
from ragelo.llm_providers.jev_provider import JevProvider, JevQuestions, JevState
from ragelo.types import LLMResponseType
from ragelo.types.configurations import TypeSafeConfiguration
from ragelo.types.formats import JevAnswer, JevResponse, LLMUsage
from ragelo.types.types import LLMProviderTypes

T_Schema = TypeVar("T_Schema", bound=BaseModel)

try:
    from typesafe_sdk import (
        Answer,
        AsyncTypeSafeClient,
        Choice,
        Noul,
        NoulAnswer,
        Question,
        RetryPolicy,
        Score,
        TypeSafeError,
    )

    _TYPESAFE_AVAILABLE = True
except ImportError:
    _TYPESAFE_AVAILABLE = False


@LLMProviderFactory.register(LLMProviderTypes.TYPESAFE)
class TypeSafeProvider(JevProvider):
    """Jev through TypeSafe's own API and SDK. Requires ``pip install 'ragelo[typesafe]'``."""

    config: TypeSafeConfiguration
    api_key_env_var: str = "TYPESAFE_API_KEY"

    def __init__(self, config: TypeSafeConfiguration, client: AsyncTypeSafeClient | None = None) -> None:
        if not _TYPESAFE_AVAILABLE:
            raise ImportError("typesafe-sdk is not installed. Install it with: pip install 'ragelo[typesafe]'")
        super().__init__(config)
        self.__client = client or AsyncTypeSafeClient(
            api_key=config.api_key.get_secret_value(),
            model=config.model,
            base_url=config.api_base,
            timeout=config.timeout,
            retry=RetryPolicy(max_retries=config.max_retries),
        )

    async def _request(self, state: JevState, questions: JevQuestions) -> LLMResponseType[JevResponse]:
        asked = {name: self.__question(question) for name, question in questions.items()}
        try:
            response = await self.__client.system_one(state=state, questions=asked)
        except TypeSafeError as e:
            raise ValueError(f"TypeSafe request failed: {e}") from e
        parsed = JevResponse(answers={name: self.__jev_answer(answer) for name, answer in response.answers.items()})
        usage = LLMUsage(
            input_tokens=response.usage.input_tokens or 0, output_tokens=response.usage.output_tokens or 0
        )
        return LLMResponseType(raw_answer=response.model_dump_json(), parsed_answer=parsed, usage=usage)

    @staticmethod
    def __question(question: dict[str, Any]) -> Question:
        """The jev evaluators call a yes/no question a boolean, as the Vercel gateway does. TypeSafe calls it a noul."""
        question_types: dict[str, type[Noul | Choice | Score]] = {
            "boolean": Noul,
            "choice": Choice,
            "score": Score,
        }
        fields = {name: value for name, value in question.items() if name != "type"}
        return question_types[question["type"]](**fields)

    @staticmethod
    def __jev_answer(answer: Answer) -> JevAnswer:
        """TypeSafe calls a yes/no question a noul, and keys the probabilities of a score by integer level."""
        if isinstance(answer, NoulAnswer):
            return JevAnswer(type="boolean", probability=answer.noul)
        return JevAnswer(
            type=answer.type,
            score=getattr(answer, "score", None),
            probabilities={str(label): probability for label, probability in answer.probabilities.items()},
            confidence=answer.confidence,
        )
