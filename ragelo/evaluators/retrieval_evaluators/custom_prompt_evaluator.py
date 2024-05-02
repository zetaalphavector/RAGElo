from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import Document, Query, RetrievalEvaluatorTypes
from ragelo.types.configurations import CustomPromptEvaluatorConfig


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.CUSTOM_PROMPT)
class CustomPromptEvaluator(BaseRetrievalEvaluator):
    config: CustomPromptEvaluatorConfig

    def __init__(
        self,
        config: CustomPromptEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        self.prompt = config.prompt

    def _build_message(self, query: Query, document: Document) -> str:
        query_metadata = self._get_usable_fields_from_metadata(
            self.prompt, query.metadata, skip_fields=[self.config.query_placeholder]
        )
        document_metadata = self._get_usable_fields_from_metadata(
            self.prompt,
            document.metadata,
            skip_fields=[self.config.document_placeholder],
        )
        formatters = {
            self.config.query_placeholder: query.query,
            self.config.document_placeholder: document.text,
            **query_metadata,
            **document_metadata,
        }

        return self.prompt.format(**formatters)
