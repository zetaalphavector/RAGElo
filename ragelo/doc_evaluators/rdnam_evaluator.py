"""Evaluator based on the paper: Paul Thomas, Seth Spielman, Nick Craswell and
Bhaskar Mitra. Large language models can accurately predict searcher preferences.
https://arxiv.org/abs/2309.10621
"""
import json
from typing import Dict

import numpy as np

from ragelo.doc_evaluators.base_doc_evaluator import (
    DocumentEvaluator,
    DocumentEvaluatorFactory,
)
from ragelo.logger import logger


@DocumentEvaluatorFactory.register("RDNAM")
class RDNAMvaluator(DocumentEvaluator):
    prompt = """{role}Given a query and a document, you must provide a score on an integer scale of 0 to 2 with the following meanings:
2 = highly relevant, very helpful for this query
1 = relevant, may be partly helpful but might contain other irrelevant content
0 = not relevant, should never be shown for this query
Assume that you are writing a report on the subject of the topic. If you would use any of the information contained in the document in such a report, mark it 1. If the document is primarily about the topic, or contains vital information about the topic, mark it 2. Otherwise, mark it 0.

# Query
A person has typed {query} into a search engine.{narrative}{description}

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

    NARRATIVE_PROMPT = " They were looking for: {narrative}"
    DESCRIPTION_PROMPT = " They were looking for: {description}"
    ASPECTS_NARRATIVE = """Measure how well the content matches a likely intent of the query (M).
    Measure how trustworthy the web page is (T)."""  # noqa: E501
    ASPECTS_EXAMPLE = """[{{"M": 2, "T": 1, "O": 1}}, {{"M": 1..."""
    DEFAULT_EXAMPLE = """[{{"O": 1}}, {{"O": 2}}, {{"O": 0..."""
    MULTILPE_PROMPT = """We asked five search engine raters to evaluate the relevance of the web page for the query.
Each rater used their own independent judgement."""  # noqa: E501

    def __init__(
        self,
        role: str = "",
        aspects: bool = False,
        multiple: bool = False,
        narrative_file: str | None = None,
        description_file: str | None = None,
        *args,
        **kwargs,
    ):
        """Initializes an evaluator based on RDNAM framweork.
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
        super().__init__(*args, **kwargs)
        self.role = role
        self.use_narratives = False
        self.use_description = False

        if narrative_file:
            self.narratives: Dict[str, str] = self._load_from_csv(narrative_file)
            self.use_narratives = True
        if description_file:
            self.descriptions: Dict[str, str] = self._load_from_csv(description_file)
            self.use_description = True

        self.aspects_prompt = self.ASPECTS_NARRATIVE if aspects else ""
        self.multiple_prompt = self.MULTILPE_PROMPT if multiple else ""
        if multiple:
            self.prompt += "\n[{{"
        else:
            self.prompt += "\n{{"
        self.multiple = multiple

    def _build_message(
        self,
        qid: str,
        did: str,
    ) -> str:
        if self.use_narratives and qid not in self.narratives:
            logger.warning(f"No narrative found for {qid}. Will not use it")
        if self.use_description and qid not in self.descriptions:
            logger.warning(f"No description found for {qid}. Will not use it")

        description_str = (
            self.DESCRIPTION_PROMPT.format(description=self.descriptions[qid])
            if self.use_description and qid in self.descriptions
            else ""
        )
        narrative_str = (
            self.NARRATIVE_PROMPT.format(narrative=self.narratives[qid])
            if self.use_narratives and qid in self.narratives
            else ""
        )
        query = self.queries[qid]
        document = self.documents[qid][did]

        example = self.ASPECTS_EXAMPLE if self.aspects_prompt else self.DEFAULT_EXAMPLE

        formatted_prompt = self.prompt.format(
            role=self.role,
            query=query,
            doc_content=document,
            narrative=narrative_str,
            description=description_str,
            aspects=self.aspects_prompt,
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
            logger.warning(f"Failed to parse answer: {answer}")
            raise ValueError
        if self.multiple:
            try:
                scores = [int(a["O"]) for a in ans]
                return int(np.mean(scores))
            except (KeyError, ValueError):
                logger.warning(f"Failed to parse answer: {answer}")
                raise ValueError
        try:
            return int(ans["O"])
        except KeyError:
            # As a last try, try to simply get whatheve is after "O"
            try:
                return int(answer.split('"O":')[1].split(",")[0])
            except IndexError:
                logger.warning(f"Failed to parse answer: {answer}")
                raise ValueError
