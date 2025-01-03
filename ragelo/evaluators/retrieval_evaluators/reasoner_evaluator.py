from __future__ import annotations

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.types.configurations import ReasonerEvaluatorConfig
from ragelo.types.evaluables import Document
from ragelo.types.formats import LLMResponseType
from ragelo.types.query import Query
from ragelo.types.types import RetrievalEvaluatorTypes


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.REASONER)
class ReasonerEvaluator(BaseRetrievalEvaluator):
    """
    A document Evaluator that only outputs the reasoning for why a document
    is relevant.
    """

    config: ReasonerEvaluatorConfig
    prompt = """
You are an expert document annotator, evaluating if a document contains relevant \
information to answer a question submitted by a user. \
Please act as an impartial relevance annotator for a search engine. \
Your goal is to evaluate the relevancy of the documents given a user question.

You should write only one sentence reasoning wether the document is relevant or not for \
the user question. A document can be:
    - Not relevant: The document is not on topic.
    - Somewhat relevant: The document is on topic but does not fully answer the \
user question.
    - Very relevant: The document is on topic and answers the user question.
    [user question]
    {query}

    [document content]
    {document}
""".strip()

    def _build_message(self, query: Query, document: Document) -> str:
        formatters = {
            self.config.query_placeholder: query.query,
            self.config.document_placeholder: document.text,
        }
        return self.prompt.format(**formatters)

    def _process_answer(self, llm_response: LLMResponseType) -> LLMResponseType:
        assert isinstance(llm_response.parsed_answer, str)
        return llm_response
