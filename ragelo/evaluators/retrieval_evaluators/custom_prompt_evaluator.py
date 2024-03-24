import json
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
        query = self.queries[qid]
        document = self.documents[qid][did]
        return self.__prompt.format(query=query.query, passage=document.text)

    def _process_answer(self, answer: str) -> str:
        return self.json_answer_parser(answer, "relevance")
