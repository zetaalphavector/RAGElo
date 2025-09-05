from __future__ import annotations

import re
from typing import Any, Optional, Type

from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ragelo.types.types import AnswerEvaluatorTypes
from ragelo.utils import string_to_template


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
    llm_response_schema: Optional[Type[BaseModel] | dict[str, Any]] = Field(
        default=None,
        description="The response schema for the LLM. If set, should be a json schema or a Pydantic BaseModel (not an instance). Otherwise, the answer will be returned as a string.",
    )
    system_prompt: Optional[Template] = Field(
        default=None,
        description="The system prompt to use for the evaluator.",
    )

    user_prompt: Optional[Template] = Field(
        default=None,
        description="The user prompt to use for the evaluator. Should contain at least a {{ query.query }} placeholder for the query's text.",
    )

    @field_validator("system_prompt", "user_prompt", mode="before")
    def check_system_prompt_and_user_prompt(cls, v: Optional[str | Template]) -> Optional[Template]:
        if isinstance(v, str):
            return string_to_template(v)
        return v

    @field_validator("user_prompt", mode="after")
    def validate_query_and_document_placeholders(cls, prompt: Template) -> Template:
        src = getattr(prompt, "_ragelo_source", None)
        placeholders = set(m.group(1) for m in re.finditer(r"{{\s*([a-zA-Z_][\w\.]*)\s*}}", src or ""))
        if "query.query" not in placeholders:
            raise ValueError("The user prompt must contain a {{query.query}} placeholder")
        return prompt
