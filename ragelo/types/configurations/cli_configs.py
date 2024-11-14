from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from ragelo.types.configurations.base_configs import BaseEvaluatorConfig
from ragelo.types.configurations.retrieval_evaluator_configs import (
    BaseRetrievalEvaluatorConfig,
    DomainExpertEvaluatorConfig,
    RDNAMEvaluatorConfig,
    ReasonerEvaluatorConfig,
)
from ragelo.types.formats import AnswerFormat


class BaseCLIConfig(BaseEvaluatorConfig):
    experiment_name: str = Field(
        default="experiment",
        description="The name of the experiment to run. This is also used as the local cache file name",
    )
    data_dir: str = Field(
        default="data",
        description="The directory where the data is stored.",
    )
    queries_csv_file: str = Field(
        default="queries.csv",
        description="The path to the queries CSV file. The file should contain at least the following columns: "
        "qid, query. Any additional columns will be considered as metadata.",
    )
    verbose: bool = Field(
        default=True,
        description="Whether or not to be verbose and print all intermediate steps.",
    )


class CLIRetrievalEvaluatorConfig(BaseCLIConfig, BaseRetrievalEvaluatorConfig):
    documents_csv_file: str = Field(
        default="documents.csv",
        description="The path to the documents CSV file. The file should contain at least the following columns: "
        "qid, did, document. Any additional columns will be considered as metadata.",
    )
    llm_answer_format: str | AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM for the retrieval evaluator.",
    )
    llm_response_schema: Type[PydanticBaseModel] | dict[str, Any] | None = Field(
        default=None,
        description=(
            "The response schema for the LLM for the retrieval evaluator. "
            "Required if the llm_answer_format is structured and recommended for JSON."
        ),
    )


class CLIDomainExpertEvaluatorConfig(CLIRetrievalEvaluatorConfig, DomainExpertEvaluatorConfig):
    pass


class CLIReasonerEvaluatorConfig(CLIRetrievalEvaluatorConfig, ReasonerEvaluatorConfig):
    pass


class CLIRDNAMEvaluatorConfig(CLIRetrievalEvaluatorConfig, RDNAMEvaluatorConfig):
    pass


class CLIConfig(BaseCLIConfig):
    documents_csv_file: str = Field(
        default="documents.csv",
        description="The path to the documents CSV file. The file should contain at least the following columns: "
        "qid, did, document. Any additional columns will be considered as metadata.",
    )
    answers_csv_file: str = Field(
        default="answers.csv",
        description="The path to the answers CSV file. The file should contain at least the following columns: "
        "qid, did, answer. Any additional columns will be considered as metadata.",
    )

    llm_answer_format_retrieval_evaluator: str | AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM for the retrieval evaluator.",
    )
    llm_response_schema_retrieval_evaluator: Type[PydanticBaseModel] | dict[str, Any] | None = Field(
        default=None,
        description=(
            "The response schema for the LLM for the retrieval evaluator. "
            "Required if the llm_answer_format is structured and recommended for JSON."
        ),
    )
    llm_answer_format_answer_evaluator: str | AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM for the answer evaluator.",
    )
    llm_response_schema_answer_evaluator: Type[PydanticBaseModel] | dict[str, Any] | None = Field(
        default=None,
        description=(
            "The response schema for the LLM for the answer evaluator. "
            "Required if the llm_answer_format is structured and recommended for JSON."
        ),
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
