from __future__ import annotations

from pydantic import BaseModel

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.llm_providers.jev_provider import JevProvider
from ragelo.types.formats import JevAnswer, JevResponse, LLMResponseType

JEV_REASONING = "Jev returns probabilities, not reasoning."


class JevEvaluatorMixin:
    """Shared by the evaluators that ask Jev typed questions instead of prompting an LLM."""

    llm_provider: BaseLLMProvider
    config: BaseModel
    unsupported_options: tuple[str, ...] = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.llm_provider, JevProvider):
            raise TypeError(f'{type(self).__name__} only works with the "typesafe" and "vercel-jev" LLM providers')
        enabled = [option for option in self.unsupported_options if getattr(self.config, option, False)]
        if enabled:
            raise ValueError(
                f"{type(self).__name__} does not support {', '.join(enabled)}: Jev generates no text and "
                "answers the same request the same way"
            )

    @staticmethod
    def _jev_answer(llm_response: LLMResponseType[BaseModel], question: str) -> JevAnswer:
        response = llm_response.parsed_answer
        if not isinstance(response, JevResponse):
            raise TypeError(f"Expected a JevResponse, got {type(response).__name__}")
        return response.answers[question]


class JevRubricEvaluatorMixin(JevEvaluatorMixin):
    """Asks Jev one question per rubric criterion, all in a single request.

    Jev cannot write a rubric, so the queries need one already, or the evaluator needs a
    `rubric_generator` backed by an LLM.
    """

    unsupported_options = (
        "graduated_scoring",
        "evidence_recall",
        "citation_quality",
        "rich_pairwise_output",
        "include_evidence_in_evaluation",
    )
