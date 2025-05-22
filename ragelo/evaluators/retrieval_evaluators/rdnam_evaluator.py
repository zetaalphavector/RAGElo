"""Evaluator based on the paper: Paul Thomas, Seth Spielman, Nick Craswell and
Bhaskar Mitra. Large language models can accurately predict searcher preferences.
https://arxiv.org/abs/2309.10621
"""

from __future__ import annotations

import textwrap

import jinja2
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
from ragelo.types.results import RetrievalEvaluatorResult
from ragelo.types.types import RetrievalEvaluatorTypes


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.RDNAM)
class RDNAMEvaluator(BaseRetrievalEvaluator):
    evaluation_prompt = jinja2.Template(
        textwrap.dedent(
            """
{%- if role %}
{{ role }}
{%- endif %}
Given a query and a document, you must provide a score on an integer scale \
of 0 to 2 with the following meanings:
2 = highly relevant, very helpful for this query
1 = relevant, may be partly helpful but might contain other irrelevant content
0 = not relevant, should never be shown for this query
Assume that you are writing a report on the subject of the topic. If you would \
use any of the information contained in the document in such a report, mark it 1. \
If the document is primarily about the topic, or contains vital information about \
the topic, mark it 2. Otherwise, mark it 0.

# Query
A person has typed {{ query }} into a search engine.
{%- if description or narrative %}
They were looking for: {{ description }}
{{ narrative }}
{%- endif %}

# Result
Consider the following document.
---BEGIN DOCUMENT CONTENT---
{{ doc_content }}
---END DOCUMENT CONTENT---

# Instructions
Split this problem into steps:
Consider the underlying intent of the search.
{%- if use_aspects %}
Measure how well the content matches a likely intent \
of the query.
Measure how trustworthy the web page is.
{%- endif %}
Consider the aspects above and relative importance of each, and decide on a final \
overall score.
{%- if use_multiple_annotators %}
We asked five search engine raters to evaluate \
the relevance of the web page for the query.
Each rater used their own independent judgement.
{%- endif %}
Produce a JSON with the scores without providing any reasoning. Example: \
{% if use_aspects %}{{"{\\"intent_match\\": 2, \\"trustworthiness\\": 1, \\"overall\\": 1}"}}{% else %}{{"{\\"overall\\": 1}"}}{% endif %}

""".strip()
        )
    )

    config: RDNAMEvaluatorConfig
    template: jinja2.Template

    def __init__(
        self,
        config: RDNAMEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        """Initializes an evaluator based on RDNAM framework."""
        super().__init__(config, llm_provider)
        self.llm_response_schema = {"overall": "An integer between 0 and 2 representing the score of the document."}

        self._role = self.config.annotator_role if self.config.annotator_role else ""

        # Schema updates based on config flags (aspects/multiple annotators)
        if self.config.use_aspects:
            self.llm_response_schema.update(
                {
                    "intent_match": "An integer between 0 and 2 representing the match of the document to the query intent.",
                    "trustworthiness": "An integer between 0 and 2 representing the trustworthiness of the document.",
                }
            )

        if self.config.use_multiple_annotators:
            base_schema = self.llm_response_schema.copy()
            self.llm_response_schema = {f"annotator_{i+1}": base_schema for i in range(5)}
            self.multiple = True  # Keep track if we need to aggregate results
        else:
            self.multiple = False

    def _build_message(self, query: Query, document: Document) -> str:
        narrative = ""
        if query.metadata:
            description = query.metadata.get("description", "")
            narrative = query.metadata.get("narrative", "")

        formatted_prompt = self.template.render(
            role=self._role,
            query=query.query,
            doc_content=document,
            description=description,
            narrative=narrative,
            use_aspects=self.config.use_aspects,
            use_multiple_annotators=self.config.use_multiple_annotators,
        )
        return formatted_prompt

    def _process_answer(self, llm_response: LLMResponseType, qid: str, did: str) -> RetrievalEvaluatorResult:
        assert isinstance(llm_response, dict)
        if not self.multiple:
            return RetrievalEvaluatorResult(qid=qid, did=did, raw_answer=llm_response, answer=llm_response)
        answer: dict[str, list[float]] = {"overall": []}
        if self.config.use_aspects:
            answer["intent_match"] = []
            answer["trustworthiness"] = []

        for v in llm_response.values():
            if self.config.use_aspects:
                answer["intent_match"].append(int(v["intent_match"]))
                answer["trustworthiness"].append(int(v["trustworthiness"]))
                answer["overall"].append(int(v["overall"]))
            else:
                answer["overall"].append(int(v["overall"]))
        if self.config.use_aspects:
            return RetrievalEvaluatorResult(
                qid=qid,
                did=did,
                raw_answer=llm_response,
                answer={k: float(np.mean(v)) for k, v in answer.items()},
            )
        return RetrievalEvaluatorResult(
            qid=qid,
            did=did,
            raw_answer=llm_response,
            answer=float(np.mean(answer["overall"])),
        )
