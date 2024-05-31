from typing import List

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.types import Document, Query, RetrievalEvaluatorTypes
from ragelo.types.configurations import ReasonerEvaluatorConfig


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.REASONER)
class ReasonerEvaluator(BaseRetrievalEvaluator):
    """
    A document Evaluator that only outputs the reasoning for why a document
    is relevant.
    """

    output_columns: List[str] = ["qid", "did", "raw_answer", "answer"]
    output_file: str = "reasonings.csv"
    config: ReasonerEvaluatorConfig
    prompt = """
You are an expert document annotator, evaluating if a document contains relevant \
information to answer a question submitted by a user. \
Please act as an impartial relevance annotator for a search engine. \
Your goal is to evaluate the relevancy of the documents given a user question.

You should write one sentence explaining why the document is relevant or not for \
the user question. A document can be:
    - Not relevant: The document is not on topic.
    - Somewhat relevant: The document is on topic but does not fully answer the \
user question.
    - Very relevant: The document is on topic and answers the user question.
    [user question]
    {query}

    [document content]
    {document}"""  # noqa: E501

    def _build_message(self, query: Query, document: Document) -> str:
        return self.prompt.format(query=query.query, document=document.text)

    def _process_answer(self, answer: str) -> str:
        return answer
