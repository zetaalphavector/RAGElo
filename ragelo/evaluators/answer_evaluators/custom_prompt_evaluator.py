from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
    BaseAnswerEvaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import AgentAnswer, AnswerEvaluatorTypes, Query
from ragelo.types.configurations import CustomPromptAnswerEvaluatorConfig


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.CUSTOM_PROMPT)
class CustomPromptEvaluator(BaseAnswerEvaluator):
    config: CustomPromptAnswerEvaluatorConfig
    output_file: str = "custom_prompt_evaluations.csv"

    def __init__(
        self,
        config: CustomPromptAnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        self.prompt = config.prompt

    def _build_message(self, query: Query, answer: AgentAnswer) -> str:
        documents = self._prepare_documents(query)
        query_metadata = self._get_usable_fields_from_metadata(
            self.prompt, query.metadata, skip_fields=[self.config.query_placeholder]
        )
        answer_metadata = self._get_usable_fields_from_metadata(
            self.prompt,
            answer.metadata,
            skip_fields=[self.config.answer_placeholder],
        )
        formatters = {
            self.config.query_placeholder: query.query,
            self.config.answer_placeholder: answer.text,
            self.config.documents_placeholder: documents,
            **query_metadata,
            **answer_metadata,
        }

        return self.prompt.format(**formatters)
