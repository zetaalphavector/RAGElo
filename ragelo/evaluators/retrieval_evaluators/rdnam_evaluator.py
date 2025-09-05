"""Evaluator based on the paper: Paul Thomas, Seth Spielman, Nick Craswell and
Bhaskar Mitra. Large language models can accurately predict searcher preferences.
https://arxiv.org/abs/2309.10621
"""

import numpy as np
from pydantic import BaseModel

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.answer_formats import (
    RDNAMAnswerEvaluatorFormat,
    RDNAMAnswerNoAspects,
    RDNAMMultipleAnnotatorsAnswer,
    RDNAMMultipleAnnotatorsAnswerNoAspects,
)
from ragelo.types.configurations import RDNAMEvaluatorConfig
from ragelo.types.evaluables import Document
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.query import Query
from ragelo.types.types import RetrievalEvaluatorTypes
from ragelo.utils import string_to_template


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.RDNAM)
class RDNAMEvaluator(BaseRetrievalEvaluator):
    config: RDNAMEvaluatorConfig
    system_prompt = string_to_template("""
        {% if annotator_role %}{{ annotator_role }} {% endif %} Given a query and a document, you must provide a score on an integer scale of 0 to 2 with the following meanings:
        2 = highly relevant, very helpful for this query
        1 = relevant, may be partly helpful but might contain other irrelevant content
        0 = not relevant, should never be shown for this query
        Assume that you are writing a report on the subject of the topic. If you would use any of the information contained in the document in such a report, mark it 1. If the document is primarily about the topic, or contains vital information about the topic, mark it 2. Otherwise, mark it 0.
        """)
    user_prompt = string_to_template("""
        # Query
        A person has typed {{ query.query }} into a search engine.
        {% if query.metadata and (query.metadata.description or query.metadata.narrative) %}
        They were looking for: {{ query.metadata.description }}
        {{ query.metadata.narrative }}
        {% endif %}

        # Result
        Consider the following document.
        ---BEGIN DOCUMENT CONTENT---
        {{ document.text }}
        ---END DOCUMENT CONTENT---

        # Instructions
        Split this problem into steps:
        Consider the underlying intent of the search.
        {%- if use_aspects %}
        Measure how well the content matches a likely intent of the query.
        Measure how trustworthy the web page is.
        {%- endif %}
        Consider the aspects above and relative importance of each, and decide on a final overall score.
        {%- if multiple %}
        We asked five search engine raters to evaluate the relevance of the web page for the query.
        Each rater used their own independent judgement.
        {%- endif %}""")

    def __init__(self, config: RDNAMEvaluatorConfig, llm_provider: BaseLLMProvider):
        """Initializes an evaluator based on RDNAM framework."""
        super().__init__(config, llm_provider)
        self._role = self.config.annotator_role if self.config.annotator_role else ""
        if self.config.use_aspects and self.config.use_multiple_annotators:
            self.config.llm_response_schema = RDNAMMultipleAnnotatorsAnswer
        elif self.config.use_aspects:
            self.config.llm_response_schema = RDNAMAnswerEvaluatorFormat
        elif self.config.use_multiple_annotators:
            self.config.llm_response_schema = RDNAMMultipleAnnotatorsAnswerNoAspects
        else:
            self.config.llm_response_schema = RDNAMAnswerNoAspects

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        context = {
            "query": query,
            "document": document,
            "annotator_role": self.config.annotator_role,
            "use_aspects": self.config.use_aspects,
            "multiple": self.config.use_multiple_annotators,
        }
        return LLMInputPrompt(
            system_prompt=self.system_prompt.render(**context),
            user_message=self.user_prompt.render(**context),
        )

    def _process_answer(self, llm_response: LLMResponseType) -> LLMResponseType:
        parsed = llm_response.parsed_answer
        assert isinstance(self.config.llm_response_schema, type(BaseModel))
        assert isinstance(parsed, self.config.llm_response_schema)

        if self.config.use_multiple_annotators:
            assert isinstance(parsed, RDNAMMultipleAnnotatorsAnswer)
            overall = float(
                np.mean(
                    [
                        parsed.annotator_1.overall,
                        parsed.annotator_2.overall,
                        parsed.annotator_3.overall,
                        parsed.annotator_4.overall,
                        parsed.annotator_5.overall,
                    ]
                )
            )
            if self.config.use_aspects:
                intent_match = float(
                    np.mean(
                        [
                            parsed.annotator_1.intent_match,
                            parsed.annotator_2.intent_match,
                            parsed.annotator_3.intent_match,
                            parsed.annotator_4.intent_match,
                            parsed.annotator_5.intent_match,
                        ]
                    )
                )
                trustworthiness = float(
                    np.mean(
                        [
                            parsed.annotator_1.trustworthiness,
                            parsed.annotator_2.trustworthiness,
                            parsed.annotator_3.trustworthiness,
                            parsed.annotator_4.trustworthiness,
                            parsed.annotator_5.trustworthiness,
                        ]
                    )
                )
            else:
                intent_match = None
                trustworthiness = None
        else:
            assert isinstance(parsed, RDNAMAnswerEvaluatorFormat) or isinstance(parsed, RDNAMAnswerNoAspects)
            overall = parsed.overall
            if self.config.use_aspects:
                assert isinstance(parsed, RDNAMAnswerEvaluatorFormat)
                intent_match = parsed.intent_match
                trustworthiness = parsed.trustworthiness
            else:
                intent_match = None
                trustworthiness = None

        response = RDNAMAnswerEvaluatorFormat(
            overall=overall,
            intent_match=intent_match,
            trustworthiness=trustworthiness,
        )

        return LLMResponseType(
            raw_answer=llm_response.raw_answer,
            parsed_answer=response,
        )
