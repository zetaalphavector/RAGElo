from __future__ import annotations

import textwrap
from typing import Any, Callable, Type

import jinja2
from pydantic import BaseModel, Field, ValidationError, model_validator

from ragelo.types.configurations.base_configs import BaseEvaluatorConfig
from ragelo.types.types import AnswerEvaluatorTypes


class BaseAnswerEvaluatorConfig(BaseEvaluatorConfig):
    pairwise: bool = Field(default=False, description="Whether or not to the evaluator is pairwise")
    has_citations: bool = Field(
        default=True,
        description=(
            "Whether or not the answers contain document citations in square brackets. "
            "If used, the document_ids in the documents and in the answers should match."
        ),
    )
    include_annotations: bool = Field(
        default=False,
        description="Whether or not to include the document relevance annotations in the prompt",
    )
    include_raw_documents: bool = Field(
        default=True,
        description="Whether or not to include the raw documents in the prompt",
    )
    document_filter: Callable[[str], bool] | None = Field(
        default=None, description="A function to filter the documents"
    )
    document_relevance_threshold: int | None = Field(
        default=None,
        description=(
            "The minimum relevance score for a document to be included in the prompt. "
            "By default, all documents are included."
        ),
    )


class PairwiseEvaluatorConfig(BaseAnswerEvaluatorConfig):
    """Configuration for the pairwise evaluator."""

    evaluator_name: AnswerEvaluatorTypes = AnswerEvaluatorTypes.PAIRWISE
    bidirectional: bool = Field(default=False, description="Whether or not to run each game in both directions")
    n_games_per_query: int = Field(default=100, description="Maximum number of games to generate for each query")
    pairwise: bool = Field(default=True, description="Whether or not to the evaluator is pairwise")
    include_annotations: bool = Field(
        default=True,
        description="Whether or not to include the document relevance annotations in the prompt",
    )
    include_raw_documents: bool = Field(
        default=False,
        description="Whether or not to include the raw documents in the prompt",
    )
    prompt: str | None = Field(
        default=None,
        description="Prompt to use for the evaluator. If not provided, a default prompt will be used",
    )
    factors: str = Field(
        default=(
            "the correctness, helpfulness, completeness, accuracy, depth, and " "level of detail of their responses"
        ),
        description=(
            "A string containing the factors to be used when evaluating an answer. "
            "If not provided, a default string will be used."
        ),
    )
    llm_response_schema: Type[BaseModel] | dict[str, Any] | None = Field(
        default={
            "agent_a_reasoning": "A string with your analysis of assistant A's answer",
            "agent_b_reasoning": "A string with your analysis of assistant B's answer",
            "verdict": (
                "The winner of the comparison. "
                "'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie"
            ),
        },
        description=(
            "The response schema for the LLM. "
            "Required if the llm_answer_format is structured and recommended for JSON."
        ),
    )
    system_prompt = jinja2.Template(
        textwrap.dedent(
            """
Please act as an impartial judge and evaluate the quality of the responses provided by by two AI assistants tasked to answer the question displayed below, based on a set of documents retrieved by a search engine.

You should choose the assistant that best answers the user question based on the set of retrieved documents, that may or not be relevant.
{% if include_annotations and include_raw_documents %}
For each reference document, you will be provided with the content of the document as well as a reasoning why the document is or is not relevant.
{%- elif include_annotations -%}
For each reference document, you will be provided with their relevance annotation.
{%- elif include_raw_documents -%}
You will be provided with the text of each reference document.
{% endif %}
{% if citations %}When citing documents, the answers cite them using square brackets.{% endif %}

Your evaluation should consider factors such as {{ factors }}.
Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only use relevant information.
Begin your evaluation by explaining how well each answer answers the user question. Then, you should compare the two responses and provide a short explanation on their differences.
After providing your explanation, output your final verdict by strictly following this format: 'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
""".strip()
        )
    )
    evaluation_prompt = jinja2.Template(
        textwrap.dedent(
            """
[User Question]
{{query.query}}
{% for key, value in (query.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}

[Reference Documents]
{% for document in documents %}
[{{ document.did }}] {% if include_raw_document and document.text is defined %} {{ document.text }} {% endif %} {% if include_annotations and document.evaluation.answer is defined%} ({{ document.evaluation.answer }}){% endif %}
{% for key, value in (document.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}
{% endfor %}

[The Start of Assistant A's Answer]
{{answer_a.text}}
{% for key, value in (answer_a.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{{answer_b.text}}
{% for key, value in (answer_b.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}
[The End of Assistant B's Answer]
""".strip()
        )
    )

    @model_validator(mode="before")
    @classmethod
    def check_annotations_or_raw_documents(cls, v):
        if v is None and cls.include_annotations is False and cls.include_raw_documents is False:
            raise ValidationError("At least one of include_annotations or include_raw_documents must be True")
        return v


class CustomPairwiseEvaluatorConfig(BaseAnswerEvaluatorConfig):
    """Configuration for a custom pairwise evaluator."""

    evaluator_name: AnswerEvaluatorTypes = AnswerEvaluatorTypes.CUSTOM_PAIRWISE
    bidirectional: bool = Field(default=False, description="Whether or not to run each game in both directions")
    n_games_per_query: int = Field(default=100, description="Maximum number of games to generate for each query")
    pairwise: bool = Field(default=True, description="Whether or not to the evaluator is pairwise")
    llm_response_schema: Type[BaseModel] | dict[str, Any] | None = Field(
        default={
            "analysis_assistant_a": "A string with your analysis of assistant A's answer",
            "analysis_assistant_b": "A string with your analysis of assistant B's answer",
            "verdict": (
                "The winner of the comparison. "
                "'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie"
            ),
        },
        description=(
            "The response schema for the LLM. "
            "Required if the llm_answer_format is structured and recommended for JSON."
        ),
    )


class CustomPromptAnswerEvaluatorConfig(BaseAnswerEvaluatorConfig):
    evaluator_name: AnswerEvaluatorTypes = AnswerEvaluatorTypes.CUSTOM_PROMPT
    prompt: str = Field(
        default="retrieved documents: {documents} query: {query} answer: {answer}",
        description=(
            "The prompt to be used to evaluate the documents. "
            "It should contain a {query} and a {document} placeholder"
        ),
    )
    include_annotations: bool = False
    include_raw_documents: bool = True


class PairwiseDomainExpertEvaluatorConfig(PairwiseEvaluatorConfig):
    evaluator_name: AnswerEvaluatorTypes = AnswerEvaluatorTypes.DOMAIN_EXPERT
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
    llm_response_schema: Type[BaseModel] | dict[str, Any] | None = Field(
        default={
            "agent_a_reasoning": "A string with your analysis of assistant A's answer",
            "agent_b_reasoning": "A string with your analysis of assistant B's answer",
            "verdict": (
                "The winner of the comparison. 'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie"
            ),
        }
    )
    include_annotations: bool = True
    include_raw_documents: bool = False
    system_prompt = jinja2.Template(
        textwrap.dedent(
            """
You are a domain expert in {{ expert_in }}.{% if company %} You work for {{ company }}.{% endif %} You are tasked with evaluating the quality of the responses provided by two AI assistants that generated answers to the question shown below, based on a set of documents retrieved by a search engine.
These assistants will be primarily used by internal users{% if company %} of {{ company }}{% endif %}{% if domain_short %} but they may also serve some of external users like {{ domain_short }}{% endif %}.

Given the original user question and a set of reference documents, you should choose the assistant that best answers the user question based on the provided documents, that may or not be relevant.

{% if include_annotations and include_raw_documents %}
For each reference document, you will be provided with the content of the document as well as a reasoning why the document is or is not relevant.
{%- elif include_annotations -%}
For each reference document, you will be provided with their relevance annotation.
{%- elif include_raw_documents -%}
You will be provided with the text of each reference document.
{% endif %}
{% if citations %}When citing documents, the answers cite them using square brackets.{% endif %}

Your evaluation should consider factors such as {{ factors }}.
Details are only useful if they answer the user question. If an answer contains non-relevant details, it should not be preferred over one that only use relevant information.
Begin your evaluation by explaining how well each answer answers the user question. Then, you should compare the two responses and provide a short explanation on their differences.
After providing your explanation, output your final verdict by strictly following this format: 'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie.
Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
""".strip()
        )
    )
    evaluation_prompt = jinja2.Template(
        textwrap.dedent(
            """
[User Question]
{{ query.query }}
{% for key, value in (query.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}

[Reference Documents]
{% for document in documents %}
[{{ document.did }}] {% if include_raw_document and document.text is defined %} {{ document.text }} {% endif %} {% if include_annotations and document.evaluation.answer is defined%} ({{ document.evaluation.answer }}){% endif %}
{% for key, value in (document.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}
{% endfor %}

[The Start of Assistant A's Answer]
{{ answer_a.text }}
{% for key, value in (answer_a.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{{ answer_b.text }}
{% for key, value in (answer_b.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}
[The End of Assistant B's Answer]
""".strip()
        )
    )


class ChatPairwiseEvaluatorConfig(PairwiseEvaluatorConfig):
    evaluator_name: AnswerEvaluatorTypes = AnswerEvaluatorTypes.CHAT_PAIRWISE
    system_prompt = jinja2.Template(
        textwrap.dedent(
            """
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants, tasked to provide answers grounded in a set of documents retrieved by a search engine in order to satisfy the user's intent displayed below.
You should choose the assistant that best satisfies the user's intent based on a set of reference documents that may or may not be relevant.
{% if include_annotations and include_raw_documents %}
For each reference document, you will be provided with the content of the document as well as a reasoning why the document is or is not relevant.
{%- elif include_annotations -%}
For each reference document, you will be provided with their relevance annotation.
{%- elif include_raw_documents -%}
You will be provided with the text of each reference document.
{% endif %}
{% if citations %}When citing documents, the answers cite them using square brackets.{% endif %}
Your evaluation should consider the evaluation objectives listed below.
If an answer contains non-relevant details, it should not be preferred over one that only use relevant information.
Begin your evaluation by examining each agent separately and explaining if each answer is useful towards satisfying the user's intent at each iteration. Then, provide a short explanation how the agent performed overall based on the evaluation objectives. 
Finally, compare the conversations of the two agents and provide a short explanation on their differences. Avoid any position biases and ensure that the order in which the conversations were presented does not influence your decision. 
Do not allow the length of the responses to influence your evaluation. Be as objective as possible.
After providing your explanation, output your final verdict by strictly following this format: 'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie."""
        )
    )
    evaluation_prompt = jinja2.Template(
        textwrap.dedent(
            """
[User Intent]
{{ query.query }}
{% for key, value in (query.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}

[Evaluation Objectives]
{{ factors }}

[Reference Documents]
{% for document in documents %}
[{{ document.did }}] {% if include_raw_document and document.text is defined %} {{ document.text }} {% endif %} {% if include_annotations and document.evaluation.answer is defined%} ({{ document.evaluation.answer }}){% endif %}
{% for key, value in (document.metadata or {}).items() %}
[{{key}}]: {{value}}
{% endfor %}
{% endfor %}

[The Start of Conversation with Assistant A]
{{ conversation_a }}
[The End of Conversation with Assistant A]

[The Start of Conversation with Assistant B]
{{ conversation_b }}
[The End of Conversation with Assistant B]
""".strip()
        )
    )
