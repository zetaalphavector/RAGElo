"""Evaluator with a domain expert persona"""

from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator, RetrievalEvaluatorFactory
from ragelo.types.configurations import DomainExpertEvaluatorConfig
from ragelo.types.evaluables import Document
from ragelo.types.formats import LLMInputPrompt
from ragelo.types.query import Query
from ragelo.types.types import RetrievalEvaluatorTypes
from ragelo.utils import string_to_template


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.DOMAIN_EXPERT)
class DomainExpertEvaluator(BaseRetrievalEvaluator):
    config: DomainExpertEvaluatorConfig
    system_prompt = string_to_template("""
        You are a domain expert in {{ expert_in }}.{% if company %} You work for {{ company }}.{% endif %} You are tasked with evaluating the performance of a retrieval system for question answering in this domain. The question answering system will be used by internal users{% if company %} of {{ company }}{% endif %}{% if domain_short %} but it also serves some of your external users like {{ domain_short }}{% endif %}. 
        These users are interested in a retrieval system that provides relevant passages based on their questions.

        ## Evaluation Guidelines
        Please think in steps about the relevance of the retrieved document given the original query. Consider the query, the document title (if available), and the document passage. Reason whether the document is not relevant to the query, somewhat relevant to the query, or highly relevant to the query.
        Use the following guidelines to reason about the relevance of the retrieved document:
        - Not Relevant:
            The document contains information that is unrelated, outdated, or completely irrelevant to the query.
            The document may contain some keywords or phrases from the query, but the context and overall meaning do not align with the query's intent.
            The document may be from a different field or time period, rendering it irrelevant to the current query.
        - Somewhat Relevant:
            The document contains some relevant information but lacks comprehensive details or context.
            The document may discuss a related topic or concept but not directly address the query.
            The information in the document is tangentially related to the query, but the primary focus remains different.
        - Highly Relevant:
            The document directly addresses the main points of the query and provides comprehensive and accurate information.
            The document may cite relevant information directly applicable to the query.
            The document may be recent and from the same field as the query, enhancing its relevance.
        General Guidelines:
            - Context Matters: Annotators should evaluate the relevance of documents within the specific context provided by the query. Understanding the 
            - nuances and domain-specific terminology is essential.
            - Content Overlap: Consider the extent of content overlap between the document and the query. Assess whether the document covers the core aspects of the query or only peripheral topics.
            - Neutrality: Base judgments solely on the content's relevance and avoid any personal opinions or biases.
            - Uncertainty: If uncertain about a relevance judgement, annotators default to a lower relevance.
        {% if extra_guidelines %}
        {%- for g in extra_guidelines %}
            - {{ g }}
        {% endfor %}
        {% endif %}
        Given the analysis above, assign a relevance score of 0, 1, or 2 to the retrieved document for this query, where:
        - 0: the document is not relevant to the query
        - 1: the document is somewhat relevant to the query
        - 2: the document is highly relevant to the query

        Respond STRICTLY as a JSON object with the following keys:
        - "reasoning": a concise explanation of your judgment
        - "score": an integer (0, 1, or 2)""")

    user_prompt = string_to_template("""
        User query:
        {{ query.query }}

        Document passage:
        {{ document.text }}""")

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        context = {
            "query": query,
            "document": document,
            "extra_guidelines": self.config.extra_guidelines or [],
            "expert_in": self.config.expert_in,
            "company": self.config.company,
            "domain_short": self.config.domain_short,
        }
        system_prompt = self.system_prompt.render(**context)
        user_message = self.user_prompt.render(**context)
        return LLMInputPrompt(system_prompt=system_prompt, user_message=user_message)
