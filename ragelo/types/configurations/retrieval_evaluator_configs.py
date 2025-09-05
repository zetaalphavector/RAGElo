from __future__ import annotations

import re
from typing import Any, Optional, Type

from jinja2 import Template
from pydantic import BaseModel, Field, field_validator

from ragelo.types.answer_formats import RetrievalAnswerEvaluatorFormat
from ragelo.types.configurations.base_configs import BaseEvaluatorConfig
from ragelo.types.types import RetrievalEvaluatorTypes
from ragelo.utils import string_to_template


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
    user_prompt: Optional[Template] = Field(
        default=None,
        description="The user prompt to use for the evaluator. Should contain at least a {{ query.query }} and a {{ document.text }} placeholder for the query and the document text.",
    )
    llm_response_schema: Optional[Type[BaseModel] | dict[str, Any]] = Field(default=RetrievalAnswerEvaluatorFormat)

    @field_validator("user_prompt", mode="after")
    def validate_user_prompt(cls, prompt: Optional[Template]) -> Optional[Template]:
        if prompt is None:
            return prompt
        src = getattr(prompt, "_ragelo_source", None)
        placeholders = set(m.group(1) for m in re.finditer(r"{{\s*([a-zA-Z_][\w\.]*)\s*}}", src or ""))
        if "query.query" not in placeholders:
            raise ValueError("The user prompt must contain a {{ query.query }} placeholder")
        if "document.text" not in placeholders:
            raise ValueError("The user prompt must contain a {{ document.text }} placeholder")
        return prompt


class ReasonerEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.REASONER


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


class CustomPromptEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.CUSTOM_PROMPT
    user_prompt: Template = Field(
        description=(
            "The user prompt to be used to evaluate the documents. "
            "It should contain at least a {{query.query}} and a {{document.text}} placeholder"
        ),
    )


class FewShotEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.FEW_SHOT

    few_shots: list[FewShotExample] = Field(
        default_factory=list,
        description="A list of few-shot examples to be used in the prompt",
    )
    few_shot_assistant_answer: Template | None = Field(
        default=None,
        description="The expected answer format from the LLM for each evaluated document "
        "It should contain at least a {{relevance}} placeholder, and, optionally, a {{reasoning}} placeholder",
    )

    @field_validator("few_shot_assistant_answer", mode="before")
    def string_to_template_few_shot_assistant_answer(cls, prompt: str | None) -> Template | None:
        if prompt is None:
            return prompt
        return string_to_template(prompt)

    @field_validator("few_shot_assistant_answer", mode="after")
    def validate_few_shot_assistant_answer(cls, prompt: Template | None) -> Template | None:
        if prompt is None:
            return prompt
        src = getattr(prompt, "_ragelo_source", None)
        placeholders = set(m.group(1) for m in re.finditer(r"{{\s*([a-zA-Z_][\w\.]*)\s*}}", src or ""))
        if "relevance" not in placeholders:
            raise ValueError("The few_shot_assistant_answer must contain a {{ relevance }} placeholder")
        return prompt


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
