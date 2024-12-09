from __future__ import annotations

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
    BaseAnswerEvaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import CustomPairwiseEvaluatorConfig
from ragelo.types.evaluables import PairwiseGame
from ragelo.types.query import Query
from ragelo.types.types import AnswerEvaluatorTypes


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.CUSTOM_PAIRWISE)
class CustomPairwiseEvaluator(BaseAnswerEvaluator):
    """A custom pairwise evaluator that allows for additional customization."""

    config: CustomPairwiseEvaluatorConfig
    system_prompt: str
    user_prompt: str

    def __init__(
        self,
        config: CustomPairwiseEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        self.system_prompt = config.system_prompt
        self.user_prompt = config.user_prompt

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> str | list[dict[str, str]]:
        system_prompt_msg = {"role": "system", "content": self.system_prompt}
        messages = [system_prompt_msg]
        documents = self._prepare_documents(query)
        query_metadata = self._get_usable_fields_from_metadata(
            self.user_prompt,
            query.metadata,
            skip_fields=[self.config.query_placeholder],
        )
        answer_a_metadata = self._get_usable_fields_from_metadata(
            self.user_prompt,
            game.agent_a_answer.metadata,
            skip_fields=[self.config.answer_placeholder],
        )
        answer_b_metadata = self._get_usable_fields_from_metadata(
            self.user_prompt,
            game.agent_b_answer.metadata,
            skip_fields=[self.config.answer_placeholder],
        )
        formatters = {
            self.config.query_placeholder: query.query,
            self.config.documents_placeholder: documents,
            "answer_a": game.agent_a_answer.text,
            "answer_b": game.agent_b_answer.text,
            **query_metadata,
            **answer_a_metadata,
            **answer_b_metadata,
        }
        user_message = self.user_prompt.format(**formatters)
        messages.append({"role": "user", "content": user_message})
        return messages
