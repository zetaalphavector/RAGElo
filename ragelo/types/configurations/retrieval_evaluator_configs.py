from __future__ import annotations

import textwrap
from typing import Any, Type

import jinja2
from pydantic import BaseModel, Field, model_validator

from ragelo.types.configurations.base_configs import BaseEvaluatorConfig
from ragelo.types.types import RetrievalEvaluatorTypes


class FewShotExample(BaseModel):
    """A few-shot example used in the few-shot retrieval evaluator.
    Args:
        passage str: The passage of the example.
        query str: The query of the example.
        relevance int: The relevance of the example.
        reasoning str: The reasoning behind the relevance
    """

    passage: str
    query: str
    relevance: int
    reasoning: str


class BaseRetrievalEvaluatorConfig(BaseEvaluatorConfig):
    evaluation_prompt: jinja2.Template = Field(
        default=jinja2.Template(
            textwrap.dedent(
                """
[USER'S QUERY]
{{query.query}}
{% for key, value in (query.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}
[END OF USER'S QUERY]
[START OF DOCUMENT]
{{document.text}}
{% for key, value in (document.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}
[END OF DOCUMENT]""".strip()
            )
        ),
        description=(
            "The prompt to be used to evaluate the documents. "
            "It should be a jinja2 template that can be rendered with a query and a document."
        ),
    )


class ReasonerEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.REASONER
    evaluation_prompt = jinja2.Template(
        textwrap.dedent(
            """
You are an expert document annotator, evaluating if a document contains relevant information to answer a question submitted by a user.
Please act as an impartial relevance annotator for a search engine. 
Your goal is to evaluate the relevancy of the documents given a user question.

You should write only one sentence reasoning wether the document is relevant or not for the user question. A document can be:
    - Not relevant: The document is not on topic.
    - Somewhat relevant: The document is on topic but does not fully answer the user question.
    - Very relevant: The document is on topic and answers the user question.

[USER QUESTION]
    {{query.query}}
    {% for key, value in (query.metadata or {}).items() %}
    [{{key}}]: {{value}}
    {% endfor %}
[END OF USER QUESTION]

[DOCUMENT CONTENT]
    {{document.text}}
    {% for key, value in (document.metadata or {}).items() %}
    [{{key}}]: {{value}}
    {% endfor %}
[END OF DOCUMENT CONTENT]""".strip()
        )
    )


class DomainExpertEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.DOMAIN_EXPERT
    expert_in: str = Field(description="What the LLM should mimic being an expert in.")
    domain_short: str | None = Field(
        default=None,
        description="A short or alternative name of the domain. (e.g., Chemistry, CS, etc.)",
    )
    company: str | None = Field(
        default=None,
        description="Name of the company or organization that the user that "
        "submitted the query works for. that the domain belongs to. "
        "(e.g.: ChemCorp, CS Inc.)",
    )
    extra_guidelines: list[str] | None = Field(
        default=None,
        description="A list of extra guidelines to be used when reasoning about the relevancy of the document.",
    )
    llm_response_schema: Type[BaseModel] | dict[str, Any] | None = Field(
        default={
            "reasoning": "The reasoning behind the relevance of the document",
            "score": (
                "An integer between 0 and 2 representing the score of the document, "
                "where 0 means the document is not relevant to the query, 1 means the document is somewhat relevant, "
                "and 2 means the document is highly relevant."
            ),
        },
    )
    system_prompt = jinja2.Template(
        textwrap.dedent(
            """
You are a domain expert in {{ expert_in }}.{% if company %} You work for {{ company }}.{% endif %} You are tasked with evaluating the performance of a retrieval system for question answering in this domain. The question answering system will be used by internal users{% if company %} of {{ company }}{% endif %}{% if domain_short %} but it also serves some of your external users like {{ domain_short }}{% endif %}. They are interested in a retrieval system that provides relevant passages based on their questions.

Given a query and a passage, you must provide a reasoning wether the document is not relevant to the query, somewhat relevant to the query or highly relevant to the query.
Consider the query, the passage title, and its content in your reasoning.
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
    - Context Matters: Annotators should evaluate the relevance of documents within the specific context provided by the query. Understanding the nuances and domain-specific terminology is essential.
    - Content Overlap: Consider the extent of content overlap between the document and the query. Assess whether the document covers the core aspects of the query or only peripheral topics.
    - Neutrality: Base judgments solely on the content's relevance and avoid any personal opinions or biases.
    - Uncertainty: If uncertain about a relevance judgement, annotators default to a lower relevance.
{% for guideline in (extra_guidelines or []) %}
    - {{ guideline }}
{% endfor %}""".strip()
        )
    )

    evaluation_prompt = jinja2.Template(
        textwrap.dedent(
            """
Given the following query and passage:

[[USER QUERY]]
{{query.query}}
{% for key, value in (query.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}

[[PASSAGE CONTENT]]
{{document.text}}
{% for key, value in (document.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}

Please think in steps about the relevance of the passage given the original query. You should reason about the relevance of the passage to the query and, after that, provide a score of 0, 1, or 2 to the passage given the particular query.
The score meaning is as follows:
- 0: indicates that the retrieved passage is not relevant to the query
- 1: the passage is somewhat relevant to the query
- 2: the passage is highly relevant to the query
""".strip()
        )
    )


class CustomPromptEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.CUSTOM_PROMPT


class FewShotEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.FEW_SHOT
    few_shots: list[FewShotExample] = Field(
        default_factory=list,
        description="A list of few-shot examples to be used in the prompt",
    )
    few_shot_user_prompt: jinja2.Template = Field(
        default=jinja2.Template(
            textwrap.dedent(
                """
[USER'S QUERY]
{{example.query}}
[END OF USER'S QUERY]
[START OF DOCUMENT]
{{example.passage}}
[END OF DOCUMENT]""".strip()
            )
        ),
        description="The individual prompt to be used to evaluate the documents.",
    )
    few_shot_assistant_answer: jinja2.Template = Field(
        default=jinja2.Template(
            textwrap.dedent(
                """
reasoning: {{example.reasoning}}
relevance: {{example.relevance}}""".strip()
            )
        ),
        description="The expected answer format from the LLM for each evaluated document "
        "It should contain a {reasoning} and a {relevance} placeholder",
    )

    @model_validator(mode="before")
    @classmethod
    def check_prompts(cls, values):
        if isinstance(values.few_shot_user_prompt, str):
            values.few_shot_user_prompt = jinja2.Template(values.few_shot_user_prompt)
        if isinstance(values.few_shot_assistant_answer, str):
            values.few_shot_assistant_answer = jinja2.Template(values.few_shot_assistant_answer)
        return values


class RDNAMEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.RDNAM
    annotator_role: str | None = Field(
        default=None,
        description="A String defining the type of user the LLM should mimic. "
        "(e.g.: 'You are a search quality rater evaluating the relevance "
        "of web pages')",
    )
    use_aspects: bool = Field(
        default=False,
        description="Should the prompt include aspects to get to the final score? "
        "If true, will prompt the LLM to compute scores for M (intent match) "
        "and T (trustworthy) for the document before computing the final score.",
    )
    use_multiple_annotators: bool = Field(
        default=False,
        description="Should the prompt ask the LLM to mimic multiple annotators?",
    )
    system_prompt = jinja2.Template(
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
""".strip()
        )
    )
    evaluation_prompt = jinja2.Template(
        textwrap.dedent(
            """
# Query
A person has typed {{ query.query }} into a search engine.
{%- if description or narrative %}
They were looking for: {% if description %}{{ description }}{% endif %}{% if narrative %}{{ narrative }}{% endif %}
{%- endif %}

# Result
Consider the following document.
---BEGIN DOCUMENT CONTENT---
{{ document.text }}
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
