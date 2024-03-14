from typing import Dict

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import Document, Query
from ragelo.types.configurations import CustomPromptEvaluatorConfig


@RetrievalEvaluatorFactory.register("custom_prompt")
class CustomPromptEvaluator(RetrievalEvaluatorFactory):
    def __init__(
        self,
        config: CustomPromptEvaluatorConfig,
        queries: Dict[str, Query],
        documents: Dict[str, Dict[str, Document]],
        llm_provider: BaseLLMProvider,
    ):
        if not queries:
            raise ValueError(
                "You are trying to use a Retrieval Evaluator without providing queries"
            )
        if not documents:
            raise ValueError(
                "You are trying to use a Retrieval Evaluator without providing documents"
            )

        self.queries = queries
        self.documents = documents
        if not config.output_file:
            self.output_file = "retrieval_evaluator.log"
        else:
            self.output_file = config.output_file
        self.llm_provider = llm_provider
        self.config = config

        self.__prompt = config.prompt

    def _build_message(self, qid: str, did: str) -> str:
        query = self.queries[qid]
        document = self.documents[qid][did]
        return self.__prompt.format(query=query.query, document=document.text)
