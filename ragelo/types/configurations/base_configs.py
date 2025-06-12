from __future__ import annotations

from typing import Any, Literal, Type

import jinja2
from pydantic import BaseModel, Field, field_validator

from ragelo.types.types import AnswerEvaluatorTypes


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
    llm_provider_name: Literal["openai", "ollama"] = Field(
        default="openai", description="The name of the LLM provider to be used."
    )
    use_progress_bar: bool = Field(
        default=True,
        description="Whether or not to show a progress bar while running the evaluations.",
    )
    n_processes: int = Field(
        default=1,
        description="The number of parallel LLM calls to use for the evaluation.",
    )


class BaseEvaluatorConfig(BaseConfig):
    evaluator_name: str | AnswerEvaluatorTypes | None = Field(
        default=None,
        description="The name of the evaluator to use.",
    )
    llm_response_schema: Type[BaseModel] | dict[str, Any] | None = Field(
        default=None,
        description="The response schema for the LLM.",
    )
    system_prompt: jinja2.Template | None = Field(
        default=None,
        description="The system prompt to be used on this evaluator.",
    )
    evaluation_prompt: jinja2.Template = Field(
        default=jinja2.Template(""),
        description="The evaluation prompt to be used on this evaluator.",
    )

    @field_validator("evaluator_name", mode="before")
    @classmethod
    def check_evaluator_exists(cls, value: str | AnswerEvaluatorTypes):
        if value.lower() not in AnswerEvaluatorTypes.__members__:
            raise ValueError(f"Evaluator {value} does not exist")
        else:
            return AnswerEvaluatorTypes(value.lower())

    @field_validator("evaluation_prompt", mode="before")
    @classmethod
    def evaluation_prompt_to_template(cls, value: str | jinja2.Template):
        if isinstance(value, str):
            return jinja2.Template(value)
        return value

    @field_validator("system_prompt", mode="before")
    @classmethod
    def system_prompt_to_template(cls, value: str | jinja2.Template | None):
        if isinstance(value, str):
            return jinja2.Template(value)
        return value

    class Config:
        arbitrary_types_allowed = True
