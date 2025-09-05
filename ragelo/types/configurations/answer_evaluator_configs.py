from __future__ import annotations

import re
from typing import Any, Callable, Optional, Type

from jinja2 import Template
from pydantic import BaseModel, Field, field_validator

from ragelo.types.answer_formats import PairWiseAnswerAnswerFormat
from ragelo.types.configurations.base_configs import BaseEvaluatorConfig
from ragelo.types.evaluables import Document
from ragelo.types.types import AnswerEvaluatorTypes


class BaseAnswerEvaluatorConfig(BaseEvaluatorConfig):
    pairwise: bool = Field(default=False, description="Whether or not to the evaluator is pairwise")
    include_relevance_score: bool = Field(
        default=False,
        description="Whether or not to include the document relevance annotations in the prompt",
    )
    include_relevance_reasoning: bool = Field(
        default=True,
        description="Whether or not to include the RetrievalEvaluator reasoning for the document relevance in the prompt.",
    )
    include_raw_documents: bool = Field(
        default=False,
        description="Whether or not to include the raw documents in the prompt",
    )
    factors: Optional[str] = Field(
        default=(
            "the correctness, helpfulness, completeness, accuracy, depth, and level of detail of their responses"
        ),
        description=("A string containing the factors to be used when evaluating an answer. "),
    )
    document_filter: Optional[Callable[[Document], bool]] = Field(
        default=None,
        description=(
            "A function to filter the documents. "
            "It should take a Document object and return a boolean indicating whether the document should be included in the prompt."
        ),
    )
    document_relevance_threshold: Optional[int] = Field(
        default=None,
        description=(
            "The minimum relevance score for a document to be included in the prompt. "
            "By default, all documents are included."
        ),
    )


class PairwiseEvaluatorConfig(BaseAnswerEvaluatorConfig):
    """Configuration for the pairwise evaluator."""

    evaluator_name: AnswerEvaluatorTypes = AnswerEvaluatorTypes.PAIRWISE
    bidirectional: bool = Field(default=True, description="Whether or not to run each game in both directions")
    n_games_per_query: int = Field(default=100, description="Maximum number of games to generate for each query")
    pairwise: bool = Field(default=True, description="Whether or not to the evaluator is pairwise")
    llm_response_schema: Optional[Type[BaseModel] | dict[str, Any]] = Field(
        default=PairWiseAnswerAnswerFormat,
        description="The response schema for the LLM.",
    )

    @field_validator("user_prompt", mode="after")
    def validate_user_prompt(cls, prompt: Template) -> Template:
        src = getattr(prompt, "_ragelo_source", None)
        placeholders = set(m.group(1) for m in re.finditer(r"{{\s*([a-zA-Z_][\w\.]*)\s*}}", src or ""))
        required = {"query.query", "game.agent_a_answer.text", "game.agent_b_answer.text"}
        missing = sorted(required - placeholders)
        if missing:
            raise ValueError(f"The user prompt is missing placeholders: {', '.join(missing)}")
        return prompt


class CustomPairwiseEvaluatorConfig(PairwiseEvaluatorConfig):
    """Configuration for a custom pairwise evaluator."""

    evaluator_name: AnswerEvaluatorTypes = AnswerEvaluatorTypes.CUSTOM_PAIRWISE
    user_prompt: Template = Field(
        description=(
            "User prompt to use for the evaluator. Should contain at least a {{ query.query }}, a {{ game.agent_a_answer.text }}, and a {{ game.agent_b_answer.text }} placeholder."
        )
    )


class CustomPromptAnswerEvaluatorConfig(BaseAnswerEvaluatorConfig):
    evaluator_name: AnswerEvaluatorTypes = AnswerEvaluatorTypes.CUSTOM_PROMPT
    system_prompt: Optional[Template] = Field(
        default_factory=lambda: Template(
            "You are a helpful assistant tasked with evaluating the correctness of answers."
        ),
        description="The system prompt to use for the evaluator.",
    )
    user_prompt: Template = Field(
        default_factory=lambda: Template(
            "retrieved documents: {% for document in documents %}{{document.text}}\n{% endfor %} query: {{query.query}} answer: {{answer.text}}"
        ),
        description=(
            "The prompt to be used to evaluate the documents. It should contain a {{query.query}} and a {{document.text}} and a {{answer.text}} placeholder"
        ),
    )

    @field_validator("user_prompt", mode="after")
    def validate_user_prompt(cls, prompt: Template) -> Template:
        src = getattr(prompt, "_ragelo_source", None)
        placeholders = set(m.group(1) for m in re.finditer(r"{{\s*([a-zA-Z_][\w\.]*)\s*}}", src or ""))
        required = {"query.query", "answer.text"}
        missing = sorted(required - placeholders)
        if missing:
            raise ValueError(f"The user prompt is missing placeholders: {', '.join(missing)}")
        return prompt


class PairwiseDomainExpertEvaluatorConfig(PairwiseEvaluatorConfig):
    evaluator_name: AnswerEvaluatorTypes = AnswerEvaluatorTypes.DOMAIN_EXPERT
    expert_in: str = Field(description="What the LLM should mimic being an expert in.")
    company: Optional[str] = Field(
        default=None,
        description="Name of the company or organization that the user that "
        "submitted the query works for. that the domain belongs to. "
        "(e.g.: ChemCorp, CS Inc.)",
    )
