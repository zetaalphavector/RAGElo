import re
from typing import Any, Callable, Type

from jinja2 import Template
from pydantic import BaseModel, Field, field_validator, model_validator

from ragelo.types.configurations.base_configs import BaseEvaluatorConfig, make_template_with_source
from ragelo.types.types import AnswerEvaluatorTypes


class BaseAnswerEvaluatorConfig(BaseEvaluatorConfig):
    pairwise: bool = Field(default=False, description="Whether or not to the evaluator is pairwise")
    document_template: Template = Field(
        default_factory=lambda: make_template_with_source("[{{ document.did }}] {{ document.annotation.answer }}"),
        description="The template to format each individual document in the prompt."
        "It should contain at least a {{ document.did }} placeholder and some other document field.",
    )
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
    use_raw_document_evaluation: bool = Field(
        default=False,
        description="Whether or not to use the raw document evaluation when templating documents in the prompt",
    )

    @field_validator("document_template", mode="after")
    def validate_document_template(cls, template: Template) -> Template:
        src = getattr(template, "_ragelo_source", None)
        placeholders = set(m.group(1) for m in re.finditer(r"{{\s*([a-zA-Z_][\w\.]*)\s*}}", src or ""))
        if "document.did" not in placeholders:
            raise ValueError("The document_template must contain a {{ document.did }} placeholder")
        if not any(p.startswith("document.") and p != "document.did" for p in placeholders):
            raise ValueError(
                "The document_template must contain at least one document field. "
                "It should contain at least a {{ document.did }} placeholder and some other document field."
            )
        return template


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
    factors: str = Field(
        default=(
            "the correctness, helpfulness, completeness, accuracy, depth, and level of detail of their responses"
        ),
        description=("A string containing the factors to be used when evaluating an answer. "),
    )
    llm_response_schema: Type[BaseModel] | dict[str, Any] | None = Field(
        default={
            "answer_a_reasoning": "A string with your analysis of assistant A's answer",
            "answer_b_reasoning": "A string with your analysis of assistant B's answer",
            "comparison_reasoning": "A string with your comparison between the two answers and their differences",
            "winner": (
                "The winner of the comparison. "
                "'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie"
            ),
        },
        description="The response schema for the LLM.",
    )

    @model_validator(mode="before")
    @classmethod
    def check_annotations_or_raw_documents(cls, v):
        if v is None and cls.include_annotations is False and cls.include_raw_documents is False:
            raise ValueError("At least one of include_annotations or include_raw_documents must be True")
        return v


class CustomPairwiseEvaluatorConfig(BaseAnswerEvaluatorConfig):
    """Configuration for a custom pairwise evaluator."""

    evaluator_name: AnswerEvaluatorTypes = AnswerEvaluatorTypes.CUSTOM_PAIRWISE
    user_prompt: Template = Field(
        description=(
            "User prompt to use for the evaluator. Should contain at least a {{ query }}, a {{ answer_a }}, and a {{ answer_b }} placeholder."
        )
    )
    bidirectional: bool = Field(default=False, description="Whether or not to run each game in both directions")
    n_games_per_query: int = Field(default=100, description="Maximum number of games to generate for each query")
    pairwise: bool = Field(default=True, description="Whether or not to the evaluator is pairwise")
    llm_response_schema: Type[BaseModel] | dict[str, Any] | None = Field(
        default={
            "analysis_assistant_a": "A string with your analysis of assistant A's answer",
            "analysis_assistant_b": "A string with your analysis of assistant B's answer",
            "differences": "A string with your comparison between the two answers and their differences",
            "winner": (
                "The winner of the comparison. "
                "'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie"
            ),
        },
        description="The response schema for the LLM. If set, should be a json schema or a Pydantic BaseModel (not an instance).",
    )

    @field_validator("user_prompt", mode="after")
    def validate_user_prompt(cls, prompt: Template) -> Template:
        src = getattr(prompt, "_ragelo_source", None)
        placeholders = set(m.group(1) for m in re.finditer(r"{{\s*([a-zA-Z_][\w\.]*)\s*}}", src or ""))
        required = {"query", "answer_a", "answer_b"}
        missing = sorted(required - placeholders)
        if missing:
            raise ValueError(f"The user prompt is missing placeholders: {', '.join(missing)}")
        return prompt


class CustomPromptAnswerEvaluatorConfig(BaseAnswerEvaluatorConfig):
    evaluator_name: AnswerEvaluatorTypes = AnswerEvaluatorTypes.CUSTOM_PROMPT
    system_prompt: Template | None = Field(
        default_factory=lambda: make_template_with_source(
            "You are a helpful assistant tasked with evaluating the correctness of answers."
        ),
        description="The system prompt to use for the evaluator.",
    )
    user_prompt: Template = Field(
        default_factory=lambda: make_template_with_source(
            "retrieved documents: {% for document in documents %}{{document.text}}\n{% endfor %} query: {{query.query}} answer: {{answer}}"
        ),
        description=(
            "The prompt to be used to evaluate the documents. It should contain a {{query.query}} and a {{document.text}} placeholder"
        ),
    )
    include_annotations: bool = False
    include_raw_documents: bool = True


class PairwiseDomainExpertEvaluatorConfig(PairwiseEvaluatorConfig):
    evaluator_name: AnswerEvaluatorTypes = AnswerEvaluatorTypes.DOMAIN_EXPERT
    expert_in: str = Field(description="What the LLM should mimic being an expert in.")
    company: str | None = Field(
        default=None,
        description="Name of the company or organization that the user that "
        "submitted the query works for. that the domain belongs to. "
        "(e.g.: ChemCorp, CS Inc.)",
    )
    include_annotations: bool = True
    include_raw_documents: bool = False
