from typing import Dict

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import Document, Query
from ragelo.types.configurations import CustomPromptEvaluatorConfig


@RetrievalEvaluatorFactory.register("custom_prompt")
class CustomPromptEvaluator(BaseRetrievalEvaluator):
    config: CustomPromptEvaluatorConfig

    def __init__(
        self,
        config: CustomPromptEvaluatorConfig,
        queries: Dict[str, Query],
        documents: Dict[str, Dict[str, Document]],
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, queries, documents, llm_provider)
        self.__prompt = config.prompt

    def _build_message(self, qid: str, did: str) -> str:
        formatters = {
            self.config.query_path: self.queries[qid],
            self.config.document_placeholder: self.documents[qid][did],
        }

        return self.__prompt.format(**formatters)

    def _process_answer(self, answer: str) -> str:
        return self.json_answer_parser(answer, "relevance")
