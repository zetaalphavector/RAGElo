"""Evaluator based on the paper: Paul Thomas, Seth Spielman, Nick Craswell and
Bhaskar Mitra. Large language models can accurately predict searcher preferences.
https://arxiv.org/abs/2309.10621
"""

import json
import logging
from typing import Dict

import numpy as np
from tenacity import RetryError

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import Document, Query
from ragelo.types.configurations import RDNAMEvaluatorConfig, RetrievalEvaluatorTypes


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

    NARRATIVE_DESCRIPTION_PROMPT = "They were looking for: {description} {narrative}"
    ASPECTS_NARRATIVE = """Measure how well the content matches a likely intent \
of the query (M).
Measure how trustworthy the web page is (T).""".strip()
    ASPECTS_EXAMPLE = """[{{"M": 2, "T": 1, "O": 1}}, {{"M": 1..."""
    DEFAULT_EXAMPLE = """[{{"O": 1}}, {{"O": 2}}, {{"O": 0..."""
    MULTIPLE_PROMPT = """We asked five search engine raters to evaluate \
the relevance of the web page for the query.
Each rater used their own independent judgement."""
    config: RDNAMEvaluatorConfig
    output_columns = ["qid", "did", "raw_answer", "answer"]
    scoring_key = "answer"

    def __init__(
        self,
        config: RDNAMEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        """Initializes an evaluator based on RDNAM framework."""
        super().__init__(config, llm_provider)

        self.__role = self.config.role if self.config.role else ""
        self.__use_narratives = False
        self.__use_description = False

        if self.config.narrative_file:
            self.__narratives: Dict[str, str] = self._load_from_csv(
                self.config.narrative_file
            )
            self.__use_narratives = True
        if self.config.description_file:
            self.descriptions: Dict[str, str] = self._load_from_csv(
                self.config.description_file
            )
            self.__use_description = True

        self.__aspects_prompt = self.ASPECTS_NARRATIVE if self.config.aspects else ""
        self.multiple_prompt = self.MULTIPLE_PROMPT if self.config.multiple else ""
        if self.config.multiple:
            self.prompt += "\n[{{"
        else:
            self.prompt += "\n{{"
        self.multiple = self.config.multiple

    def evaluate_single_sample(
        self, query: Query, document: Document
    ) -> Dict[str, str]:
        """Evaluates a single query-document pair. Returns the raw answer and the processed answer."""
        qid = query.qid
        did = document.did
        message = self._build_message(query.query, document.text, qid, did)
        try:
            raw_answer = self.llm_provider(message)
        except RetryError as e:
            logging.warning(f"Failed to FETCH answers for {qid} {did}")
            raise e
        try:
            answer = self._process_answer(raw_answer)
        except ValueError as e:
            logging.warning(f"Failed to PARSE answer for {qid} {did}")
            raise e
        return {
            "qid": qid,
            "did": did,
            "raw_answer": raw_answer,
            "answer": answer,
        }

    def _build_message(self, query: str, document: str, qid: str, did: str) -> str:
        if self.__use_narratives and qid not in self.__narratives:
            logging.warning(f"No narrative found for {qid}. Will not use it")
        if self.__use_description and qid not in self.descriptions:
            logging.warning(f"No description found for {qid}. Will not use it")

        narrative = (
            self.__narratives[qid]
            if qid in self.__narratives and self.__use_narratives
            else ""
        )
        description = (
            self.descriptions[qid]
            if qid in self.descriptions and self.__use_description
            else ""
        )
        narrative_description_str = self.NARRATIVE_DESCRIPTION_PROMPT.format(
            narrative=narrative, description=description
        )

        example = (
            self.ASPECTS_EXAMPLE if self.__aspects_prompt else self.DEFAULT_EXAMPLE
        )

        formatted_prompt = self.prompt.format(
            role=self.__role,
            query=query,
            doc_content=document,
            narrative_description=narrative_description_str,
            aspects=self.__aspects_prompt,
            multiple=self.multiple_prompt,
            example=example,
        )
        return formatted_prompt

    def _process_answer(self, answer: str) -> int:
        if self.multiple:
            answer = "[{" + answer
        else:
            answer = "{" + answer
        try:
            ans = json.loads(answer)
        except json.decoder.JSONDecodeError:
            raise ValueError(f"Failed to parse answer: {answer}")
        if self.multiple:
            try:
                scores = [int(a["O"]) for a in ans]
                return int(np.mean(scores))
            except (KeyError, ValueError):
                raise ValueError(f"Failed to parse answer: {answer}")
        try:
            return int(ans["O"])
        except KeyError:
            # As a last try, try to simply get whatever is after "O"
            try:
                return int(answer.split('"O":')[1].split(",")[0])
            except IndexError:
                raise ValueError(f"Failed to parse answer: {answer}")
