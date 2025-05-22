from __future__ import annotations

import textwrap
from typing import Any, Type

import jinja2
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from ragelo.logger import logger
from ragelo.types.configurations.base_configs import AnswerFormat, BaseEvaluatorConfig
from ragelo.types.pydantic_models import BaseModel, post_validator, validator
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
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.CUSTOM_PROMPT
    evaluation_prompt: jinja2.Template = Field(
        default=jinja2.Template(
            textwrap.dedent("""[USER'S QUERY]
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
            [END OF DOCUMENT]""")
        ),
        description=(
            "The prompt to be used to evaluate the documents. "
            "It should be a jinja2 template that can be rendered with a query and a document."
        ),
    )

    @post_validator
    @classmethod
    def check_prompts(cls, values):
        if isinstance(values.evaluation_prompt, str):
            values.evaluation_prompt = jinja2.Template(values.evaluation_prompt)
        if isinstance(values.system_prompt, str):
            values.system_prompt = jinja2.Template(values.system_prompt)
        return values


class ReasonerEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.REASONER
    llm_answer_format: AnswerFormat = Field(
        default=AnswerFormat.TEXT,
        description="The format of the answer returned by the LLM",
    )

    @post_validator
    @classmethod
    def check_answer_format(cls, values):
        if values.llm_answer_format != AnswerFormat.TEXT:
            logger.warning("We are using the Reasoner Evaluator config. Forcing the LLM answer format to TEXT.")
            values.llm_answer_format = AnswerFormat.TEXT
        return values


class DomainExpertEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.DOMAIN_EXPERT
    expert_in: str = Field(description="What the LLM should mimic being an expert in.")
    domain_short: str | None = Field(
        default=None,
        description="A short or alternative name of the domain. " "(e.g., Chemistry, CS, etc.)",
    )
    company: str | None = Field(
        default=None,
        description="Name of the company or organization that the user that "
        "submitted the query works for. that the domain belongs to. "
        "(e.g.: ChemCorp, CS Inc.)",
    )
    extra_guidelines: list[str] | None = Field(
        default=None,
        description="A list of extra guidelines to be used when reasoning about the " "relevancy of the document.",
    )
    llm_answer_format: AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM",
    )
    llm_response_schema: Type[PydanticBaseModel] | dict[str, Any] | None = Field(
        default={
            "reasoning": "The reasoning behind the relevance of the document",
            "score": (
                "An integer between 0 and 2 representing the score of the document, "
                "where 0 means the document is not relevant to the query, 1 means the document is somewhat relevant, "
                "and 2 means the document is highly relevant."
            ),
        },
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
            textwrap.dedent("""[USER'S QUERY]
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
            [END OF DOCUMENT]""")
        ),
        description=(
            "The individual prompt to be used to evaluate the documents. "
            "It should contain a {query} and a {passage} placeholder"
        ),
    )
    few_shot_assistant_answer: jinja2.Template = Field(
        default=jinja2.Template(
            textwrap.dedent("""
            reasoning: {{reasoning}}
            relevance: {{relevance}}
            """)
        ),
        description="The expected answer format from the LLM for each evaluated document "
        "It should contain a {reasoning} and a {relevance} placeholder",
    )

    @validator
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
    llm_answer_format: AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM",
    )

    @post_validator
    @classmethod
    def check_answer_format(cls, values):
        if values.llm_answer_format != AnswerFormat.JSON:
            logger.warning("We are using the RDNAM Evaluator config. Forcing the LLM answer format to JSON.")
            values.llm_answer_format = AnswerFormat.JSON
        return values
