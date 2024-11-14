from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from ragelo.types.formats import AnswerFormat
from ragelo.types.pydantic_models import BaseModel


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


class BaseEvaluatorConfig(BaseConfig):
    query_placeholder: str = Field(
        default="query",
        description="The placeholder for the query in the prompt.",
    )
    llm_answer_format: str | AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM.",
    )
    llm_response_schema: Type[PydanticBaseModel] | dict[str, Any] | None = Field(
        default=None,
        description="The response schema for the LLM. Required if the llm_answer_format is structured and recommended for JSON.",
    )
    n_processes: int = Field(
        default=1,
        description="The number of parallel LLM calls to use for the evaluation.",
    )


class AllConfig(BaseEvaluatorConfig):
    retrieval_evaluator_name: str = Field(
        default="reasoner",
        description="The name of the retrieval evaluator to use",
    )
    answer_evaluator_name: str = Field(
        default="pairwise",
        description="The name of the answer evaluator to use",
    )
    answer_ranker_name: str = Field(default="elo", description="The name of the answer ranker to use")
    llm_answer_format_retrieval_evaluator: str | AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM for the retrieval evaluator.",
    )
    llm_response_schema_retrieval_evaluator: Type[PydanticBaseModel] | dict[str, Any] | None = Field(
        default=None,
        description="The response schema for the LLM for the retrieval evaluator. Required if the llm_answer_format is structured and recommended for JSON.",
    )
    llm_answer_format_answer_evaluator: str | AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM for the answer evaluator.",
    )
    llm_response_schema_answer_evaluator: Type[PydanticBaseModel] | dict[str, Any] | None = Field(
        default=None,
        description="The response schema for the LLM for the answer evaluator. Required if the llm_answer_format is structured and recommended for JSON.",
    )
    scoring_key_retrieval_evaluator: str = Field(
        default="answer",
        description="When using answer_format=json, the key to extract from the answer for the retrieval evaluator.",
    )
    k: int = Field(default=100, description="Number of pairwise games to generate")
    initial_score: int = Field(default=1000, description="The initial Elo score for each agent")
    elo_k: int = Field(default=32, description="The K factor for the Elo ranking algorithm")
    bidirectional: bool = Field(default=False, description="Wether or not to run each game in both directions")
    model: str = Field(default="gpt-4o-mini", description="The model to use for the LLM")
    rich_print: bool = Field(default=True, description="Use rich to print colorful outputs.")
