from __future__ import annotations

from typing import Optional

from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ragelo.types.results import EvaluationAnswer
from ragelo.types.types import AnswerEvaluatorTypes
from ragelo.utils import get_placeholders_and_tags, string_to_template


class BaseConfig(BaseModel):
    force: bool = Field(
        default=False,
        description="Force the execution of the commands and overwrite any existing files.",
    )
    rich_print: bool = Field(default=False, description="Use rich to print colorful outputs.")
    verbose: bool = Field(
        default=False,
        description="Whether or not to be verbose and print all intermediate steps.",
    )
    llm_provider_name: str = Field(default="openai", description="The name of the LLM provider to be used.")
    use_progress_bar: bool = Field(
        default=True,
        description="Whether or not to show a progress bar while running the evaluations.",
    )
    n_processes: int = Field(
        default=1,
        description="The number of parallel LLM calls to use for the evaluation.",
    )


class BaseEvaluatorConfig(BaseConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    evaluator_name: Optional[str | AnswerEvaluatorTypes] = Field(
        default=None,
        description="The name of the evaluator to use.",
    )

    system_prompt: Optional[Template | str] = Field(
        default=None,
        description="The system prompt to use for the evaluator.",
    )
    user_prompt: Optional[Template] = Field(
        default=None,
        description=(
            "The user prompt to use for the evaluator. Should contain at least "
            "a {{ query.query }} placeholder for the query's text."
        ),
    )

    result_type: type[EvaluationAnswer] | None = Field(
        default=None,
        description=(
            "Override the result type that the Evaluator will return. Most evaluators have a default result type."
        ),
    )

    @field_validator("system_prompt", "user_prompt", mode="before")
    def check_system_prompt_and_user_prompt(cls, v: Optional[str | Template]) -> Optional[Template]:
        if isinstance(v, str):
            return string_to_template(v)
        return v

    @field_validator("user_prompt", mode="after")
    def validate_query_and_document_placeholders(cls, prompt: Template) -> Template:
        placeholders = get_placeholders_and_tags(prompt)
        if "query.query" not in placeholders:
            raise ValueError("The user prompt must contain a {{query.query}} placeholder")
        return prompt
