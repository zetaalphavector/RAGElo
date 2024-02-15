"""Evaluator based on the paper: Paul Thomas, Seth Spielman, Nick Craswell and
Bhaskar Mitra. Large language models can accurately predict searcher preferences.
https://arxiv.org/abs/2309.10621
"""

import json
import logging
from typing import Dict

import numpy as np

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import Document, Query
from ragelo.types.configurations import RetrievalEvaluatorConfig


@RetrievalEvaluatorFactory.register("RDNAM")
class RDNAMEvaluator(BaseRetrievalEvaluator):
    prompt = """{role}Given a query and a document, you must provide a score on an integer scale of 0 to 2 with the following meanings:
2 = highly relevant, very helpful for this query
1 = relevant, may be partly helpful but might contain other irrelevant content
0 = not relevant, should never be shown for this query
Assume that you are writing a report on the subject of the topic. If you would use any of the information contained in the document in such a report, mark it 1. If the document is primarily about the topic, or contains vital information about the topic, mark it 2. Otherwise, mark it 0.

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
Consider the aspects above and relative importance of each, and decide on a final score (O).
{multiple}
Produce a JSON array of scores without providing any reasoning. Example: {example}

# Results
"""  # noqa: E501

    NARRATIVE_DESCRIPTION_PROMPT = "They were looking for: {description} {narrative}"
    ASPECTS_NARRATIVE = """Measure how well the content matches a likely intent of the query (M).
    Measure how trustworthy the web page is (T)."""  # noqa: E501
    ASPECTS_EXAMPLE = """[{{"M": 2, "T": 1, "O": 1}}, {{"M": 1..."""
    DEFAULT_EXAMPLE = """[{{"O": 1}}, {{"O": 2}}, {{"O": 0..."""
    MULTIPLE_PROMPT = """We asked five search engine raters to evaluate the relevance of the web page for the query.
Each rater used their own independent judgement."""  # noqa: E501

    def __init__(
        self,
        config: RetrievalEvaluatorConfig,
        queries: Dict[str, Query],
        documents: Dict[str, Dict[str, Document]],
        llm_provider: BaseLLMProvider,
    ):
        """Initializes an evaluator based on RDNAM framework.
        Args:
            role: A String defining the type of user the LLM should mimic
                (e.g.: "You are a search quality rater evaluating
                the relevance of web pages")
            description: Will a description of the task be provided to the LLM?
            narrative: Will a narrative of the task be provided to the LLM?
            aspects: Should the prompt include aspects to get tot he final score?
                If true, will prompt the LLM to compute scores for M (intent match)
                and T (trustworthy) for the document before computing the final score.
            multiple: Should the prompt ask the LLM to mimic multiple annotators?
        """
        super().__init__(config, queries, documents, llm_provider)
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

    def _build_message(
        self,
        qid: str,
        did: str,
    ) -> str:
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
        query = self.queries[qid]
        document = self.documents[qid][did]

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
