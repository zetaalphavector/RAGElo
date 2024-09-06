from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
    BaseAnswerEvaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import AnswerEvaluatorTypes, PairwiseGame, Query
from ragelo.types.configurations import PairwiseEvaluatorConfig


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.CUSTOM_PROMPT_PAIRWISE)
class CustomPromptPairwiseEvaluator(BaseAnswerEvaluator):
    config: PairwiseEvaluatorConfig
    output_file: str = "custom_prompt_pairwise_evaluations.csv"

    def __init__(
        self,
        config: PairwiseEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        assert (
            config.prompt is not None
        ), "Prompt is required for CustomPromptAnswerEvaluator"
        super().__init__(config, llm_provider)
        self.prompt = config.prompt

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> str:
        documents = self._prepare_documents(query)
        query_metadata = self._get_usable_fields_from_metadata(
            self.prompt, query.metadata, skip_fields=[self.config.query_placeholder]
        )
        answer_a_metadata = self._get_usable_fields_from_metadata(
            self.prompt,
            game.agent_a_answer.metadata,
            skip_fields=[self.config.answer_placeholder],
        )
        answer_b_metadata = self._get_usable_fields_from_metadata(
            self.prompt,
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
        return self.prompt.format(**formatters)
