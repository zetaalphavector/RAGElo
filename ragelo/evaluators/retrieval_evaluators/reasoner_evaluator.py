from __future__ import annotations

import textwrap

import jinja2

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.types.configurations import ReasonerEvaluatorConfig
from ragelo.types.formats import LLMResponseType
from ragelo.types.results import RetrievalEvaluatorResult
from ragelo.types.types import RetrievalEvaluatorTypes


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.REASONER)
class ReasonerEvaluator(BaseRetrievalEvaluator):
    """
    A document Evaluator that only outputs the reasoning for why a document
    is relevant.
    """

    config: ReasonerEvaluatorConfig
    evaluation_prompt = jinja2.Template(
        textwrap.dedent(
            """
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
    {{query.query}}
    {% for key, value in (query.metadata or {}).items() %}
    [{{key}}]: {{value}}
    {% endfor %}


    [document content]
    {{document.text}}
    {% for key, value in (document.metadata or {}).items() %}
    [{{key}}]: {{value}}
    {% endfor %}"""
        )
    )

    def _process_answer(self, llm_response: LLMResponseType, qid: str, did: str) -> RetrievalEvaluatorResult:
        assert isinstance(llm_response, str)
        return RetrievalEvaluatorResult(
            qid=qid,
            did=did,
            raw_answer=llm_response,
            answer=llm_response,
        )
