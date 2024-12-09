"""Evaluator based on the paper: Paul Thomas, Seth Spielman, Nick Craswell and
Bhaskar Mitra. Large language models can accurately predict searcher preferences.
https://arxiv.org/abs/2309.10621
"""

from __future__ import annotations

import numpy as np

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import RDNAMEvaluatorConfig
from ragelo.types.evaluables import Document
from ragelo.types.formats import LLMResponseType
from ragelo.types.query import Query
from ragelo.types.types import RetrievalEvaluatorTypes


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.RDNAM)
class RDNAMEvaluator(BaseRetrievalEvaluator):
    prompt = """
{role}Given a query and a document, you must provide a score on an integer scale \
of 0 to 2 with the following meanings:
2 = highly relevant, very helpful for this query
1 = relevant, may be partly helpful but might contain other irrelevant content
0 = not relevant, should never be shown for this query
Assume that you are writing a report on the subject of the topic. If you would \
use any of the information contained in the document in such a report, mark it 1. \
If the document is primarily about the topic, or contains vital information about \
the topic, mark it 2. Otherwise, mark it 0.

# Query
A person has typed {query} into a search engine.
{narrative_description}

# Result
Consider the following document.
---BEGIN DOCUMENT CONTENT---
{doc_content}
---END DOCUMENT CONTENT---

# Instructions
Split this problem into steps:
Consider the underlying intent of the search.
{aspects}
Consider the aspects above and relative importance of each, and decide on a final \
overall score.
{multiple}
Produce a JSON with the scores without providing any reasoning. Example: {example}

""".strip()

    NARRATIVE_DESCRIPTION_PROMPT = "They were looking for: {description}\n{narrative}"
    ASPECTS_NARRATIVE = """Measure how well the content matches a likely intent \
of the query.
Measure how trustworthy the web page is.""".strip()

    ASPECTS_EXAMPLE = """{"intent_match": 2, "trustworthiness": 1, "overall": 1}"""
    DEFAULT_EXAMPLE = """{"overall": 1}"""
    MULTIPLE_PROMPT = """We asked five search engine raters to evaluate \
the relevance of the web page for the query.
Each rater used their own independent judgement."""
    config: RDNAMEvaluatorConfig

    def __init__(
        self,
        config: RDNAMEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        """Initializes an evaluator based on RDNAM framework."""
        super().__init__(config, llm_provider)
        self.config.llm_response_schema = {
            "overall": "An integer between 0 and 2 representing the score of the document."
        }

        self._role = self.config.annotator_role if self.config.annotator_role else ""

        if self.config.use_aspects:
            self._aspects_prompt = self.ASPECTS_NARRATIVE
            self.config.llm_response_schema["intent_match"] = (
                "An integer between 0 and 2 representing the match of the document to the query intent."
            )
            self.config.llm_response_schema["trustworthiness"] = (
                "An integer between 0 and 2 representing the trustworthiness of the document."
            )
        else:
            self._aspects_prompt = ""
        if self.config.use_multiple_annotators:
            self._multiple_prompt = self.MULTIPLE_PROMPT
            self.config.llm_response_schema = {
                "annotator_1": self.config.llm_response_schema,
                "annotator_2": self.config.llm_response_schema,
                "annotator_3": self.config.llm_response_schema,
                "annotator_4": self.config.llm_response_schema,
                "annotator_5": self.config.llm_response_schema,
            }
            self.multiple = True
        else:
            self._multiple_prompt = ""
            self.multiple = False

    def _build_message(self, query: Query, document: Document) -> str:
        narrative_description_str = ""
        if query.metadata:
            description = query.metadata.get("description", "")
            narrative = query.metadata.get("narrative", "")
            if narrative or description:
                narrative_description_str = self.NARRATIVE_DESCRIPTION_PROMPT.format(
                    narrative=narrative, description=description
                )

        example = self.ASPECTS_EXAMPLE if self._aspects_prompt else self.DEFAULT_EXAMPLE

        formatted_prompt = self.prompt.format(
            role=self._role,
            query=query.query,
            doc_content=document,
            narrative_description=narrative_description_str,
            aspects=self._aspects_prompt,
            multiple=self._multiple_prompt,
            example=example,
        )
        return formatted_prompt

    def _process_answer(self, llm_response: LLMResponseType) -> LLMResponseType:
        assert isinstance(llm_response.parsed_answer, dict)
        if not self.multiple:
            return llm_response
        answer: dict[str, list[float]] = {"overall": []}
        if self.config.use_aspects:
            answer["intent_match"] = []
            answer["trustworthiness"] = []

        for v in llm_response.parsed_answer.values():
            if self.config.use_aspects:
                answer["intent_match"].append(int(v["intent_match"]))
                answer["trustworthiness"].append(int(v["trustworthiness"]))
                answer["overall"].append(int(v["overall"]))
            else:
                answer["overall"].append(int(v["overall"]))
        if self.config.use_aspects:
            return LLMResponseType(
                raw_answer=llm_response.raw_answer,
                parsed_answer={k: float(np.mean(v)) for k, v in answer.items()},
            )
        return LLMResponseType(
            raw_answer=llm_response.raw_answer,
            parsed_answer=float(np.mean(answer["overall"])),
        )
