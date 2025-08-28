from typing import Any, Type

from pydantic import BaseModel, Field

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
    evaluator_name: str | AnswerEvaluatorTypes | None = Field(
        default=None,
        description="The name of the evaluator to use.",
    )
    query_placeholder: str = Field(
        default="query",
        description="The placeholder for the query in the prompt.",
    )
    llm_response_schema: Type[BaseModel] | dict[str, Any] | None = Field(
        default=None,
        description="The response schema for the LLM. If set, should be a json schema or a Pydantic BaseModel (not an instance). Otherwise, the answer will be returned as a string.",
    )
