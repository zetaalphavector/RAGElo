"""Evaluator based on the paper: Paul Thomas, Seth Spielman, Nick Craswell and
Bhaskar Mitra. Large language models can accurately predict searcher preferences.
https://arxiv.org/abs/2309.10621
"""

from typing import Any

import numpy as np

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import RDNAMEvaluatorConfig
from ragelo.types.evaluables import Document
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
score (O).
{multiple}
Produce a JSON array of scores without providing any reasoning. Example: {example}

# Results
""".strip()

    NARRATIVE_DESCRIPTION_PROMPT = "They were looking for: {description}\n{narrative}"
    ASPECTS_NARRATIVE = """Measure how well the content matches a likely intent \
of the query (M).
Measure how trustworthy the web page is (T).""".strip()
    ASPECTS_EXAMPLE = """[{"M": 2, "T": 1, "O": 1}, {"M": 1..."""
    DEFAULT_EXAMPLE = """[{"O": 1}, {"O": 2}, {"O": 0..."""
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
            "O": "An integer between 0 and 2 representing the score of the document."
        }

        self.__role = self.config.annotator_role if self.config.annotator_role else ""

        if self.config.use_aspects:
            self.__aspects_prompt = self.ASPECTS_NARRATIVE
            self.config.llm_response_schema["M"] = (
                "An integer between 0 and 2 representing the match of the document to the query intent."
            )
            self.config.llm_response_schema["T"] = (
                "An integer between 0 and 2 representing the trustworthiness of the document."
            )
        else:
            self.__aspects_prompt = ""
        if self.config.use_multiple_annotators:
            self.__multiple_prompt = self.MULTIPLE_PROMPT
            self.config.llm_response_schema = {
                "annotator_1": self.config.llm_response_schema,
                "annotator_2": self.config.llm_response_schema,
                "annotator_3": self.config.llm_response_schema,
                "annotator_4": self.config.llm_response_schema,
                "annotator_5": self.config.llm_response_schema,
            }
            self.multiple = True
        else:
            self.__multiple_prompt = ""
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

        example = (
            self.ASPECTS_EXAMPLE if self.__aspects_prompt else self.DEFAULT_EXAMPLE
        )

        formatted_prompt = self.prompt.format(
            role=self.__role,
            query=query.query,
            doc_content=document,
            narrative_description=narrative_description_str,
            aspects=self.__aspects_prompt,
            multiple=self.__multiple_prompt,
            example=example,
        )
        return formatted_prompt

    def _process_answer(
        self, answer: dict[str, Any]
    ) -> int | float | dict[str, int | float]:
        scores = {"intent_match": [], "trustworthiness": [], "overall": []}
        if self.multiple:
            for v in answer.values():
                if self.config.use_aspects:
                    scores["intent_match"].append(int(v["M"]))
                    scores["trustworthiness"].append(int(v["T"]))
                    scores["overall"].append(int(v["O"]))
                else:
                    scores["overall"].append(int(v["O"]))
            if self.config.use_aspects:
                return {k: float(np.mean(v)) for k, v in scores.items()}
            return float(np.mean(scores["overall"]))
        if self.config.use_aspects:
            return {
                "intent_match": int(answer["M"]),
                "trustworthiness": int(answer["T"]),
                "overall": int(answer["O"]),
            }
        return int(answer["O"])
