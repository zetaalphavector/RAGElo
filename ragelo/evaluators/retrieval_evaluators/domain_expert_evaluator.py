"""Evaluator with a domain expert persona"""

from __future__ import annotations

from ragelo.evaluators.retrieval_evaluators import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
from ragelo.types.configurations import DomainExpertEvaluatorConfig
from ragelo.types.evaluables import Document
from ragelo.types.formats import AnswerFormat, LLMResponseType
from ragelo.types.query import Query
from ragelo.types.types import RetrievalEvaluatorTypes


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.DOMAIN_EXPERT)
class DomainExpertEvaluator(BaseRetrievalEvaluator):
    system_prompt = """
You are a domain expert in {expert_in}.{company_prompt_1} You are tasked \
with evaluating the performance of a retrieval system for question \
answering in this domain. The question answering system will be used \
by internal users{company_prompt_2}{domain_short}. They are interested \
in a retrieval system that provides relevant passages based on their questions.

Given a query and a passage, you must provide a reasoning wether the document is not \
relevant to the query, somewhat relevant to the query or highly relevant to the query. \
Consider the query, the passage title, and its content in your reasoning. \
Use the following guidelines to reason about the relevance of the retrieved document:
- Not Relevant:
    The document contains information that is unrelated, outdated, or completely \
irrelevant to the query.
    The document may contain some keywords or phrases from the query, but the context \
and overall meaning do not align with the query's intent.
    The document may be from a different field or time period, rendering it irrelevant \
to the current query.
- Somewhat Relevant:
    The document contains some relevant information but lacks comprehensive details \
or context.
    The document may discuss a related topic or concept but not directly address the \
query.
    The information in the document is tangentially related to the query, but the \
primary focus remains different.
- Highly Relevant:
    The document directly addresses the main points of the query and provides \
comprehensive and accurate information.
    The document may cite relevant information directly applicable to the query.
    The document may be recent and from the same field as the query, enhancing its \
relevance.
General Guidelines:
    - Context Matters: Annotators should evaluate the relevance of documents within \
the specific context provided by the query. Understanding the nuances \
and domain-specific terminology is essential.
    - Content Overlap: Consider the extent of content overlap between the document\
and the query. Assess whether the document covers the core aspects of the query \
or only peripheral topics.
    - Neutrality: Base judgments solely on the content's relevance and avoid any \
personal opinions or biases.
    - Uncertainty: If uncertain about a relevance judgement, annotators default to a \
lower relevance.
    {extra_guidelines}
""".strip()

    user_prompt = """ 
Given the following query and passage:

[[USER QUERY]]
{query}

[[PASSAGE CONTENT]]
{doc_content}

Please think in steps about the relevance of the passage given the \
original query. You should reason about the relevance of the passage to the query and, \
after that, provide a score of 0, 1, or 2 to the passage given the \
particular query. The score meaning is as follows:
- 0: indicates that the retrieved passage is not relevant to the query
- 1: the passage is somewhat relevant to the query
- 2: the passage is highly relevant to the query
""".strip()

    COMPANY_PROMPT_1 = " You work for {company}."
    COMPANY_PROMPT_2 = " of {company}"
    DOMAIN_SHORT = " but it also serves some of your external users like {domain_short}"
    config: DomainExpertEvaluatorConfig

    def __init__(
        self,
        config: DomainExpertEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        extra_guidelines = self.config.extra_guidelines if self.config.extra_guidelines else []
        guidelines = "\n".join([f"- {guideline}" for guideline in extra_guidelines])
        if self.config.llm_answer_format != AnswerFormat.JSON:
            logger.warning("We are using the Domain Expert Evaluator config. Forcing the LLM answer format to JSON.")
            self.config.llm_answer_format = AnswerFormat.JSON
        self.expert_in = self.config.expert_in
        self.domain_short = f" {self.config.domain_short}" if self.config.domain_short else ""
        self.system_prompt = self.system_prompt.format(
            expert_in=self.expert_in,
            company_prompt_1=(
                self.COMPANY_PROMPT_1.format(company=self.config.company) if self.config.company else ""
            ),
            company_prompt_2=(
                self.COMPANY_PROMPT_2.format(company=self.config.company) if self.config.company else ""
            ),
            domain_short=(
                self.DOMAIN_SHORT.format(domain_short=self.config.domain_short) if self.config.domain_short else ""
            ),
            extra_guidelines=guidelines,
        )

    def _build_message(self, query: Query, document: Document) -> str:
        user_prompt = self.user_prompt.format(
            query=query.query,
            doc_content=document.text,
        )
        return user_prompt

    def _process_answer(self, llm_response: LLMResponseType) -> LLMResponseType:
        assert isinstance(llm_response.parsed_answer, dict)
        return LLMResponseType(
            raw_answer=llm_response.raw_answer,
            parsed_answer=llm_response.parsed_answer["score"],
        )
