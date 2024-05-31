from importlib import metadata
from typing import List, Optional, Union

from pydantic import Field

from ragelo.types.types import AnswerFormat, BaseModel

_PYDANTIC_MAJOR_VERSION: int = int(metadata.version("pydantic").split(".")[0])


class BaseConfig(BaseModel):
    force: bool = Field(
        default=False,
        description="Force the execution of the commands and overwrite any existing files",
    )
    rich_print: bool = Field(
        default=False, description="Use rich to print colorful outputs"
    )
    verbose: bool = Field(
        default=False,
        description="Whether or not to be verbose and print all intermediate steps",
    )
    credentials_file: Optional[str] = Field(
        default=None,
        description="Path to a txt file with the credentials for the different LLM providers",
    )
    data_path: str = Field(default="data/", description="Path to the data folder")

    llm_provider: str = Field(
        default="openai", description="The name of the LLM provider"
    )
    write_output: bool = Field(
        default=True, description="Whether or not to write the output to a file"
    )
    use_progress_bar: bool = Field(
        default=True, description="Whether or not to show a progress bar"
    )


class BaseEvaluatorConfig(BaseConfig):
    query_path: str = Field(
        default="queries.csv", description="Path to the queries file"
    )
    documents_path: str = Field(
        default="documents.csv", description="Path to the documents file"
    )
    answers_path: str = Field(
        default="answers.csv", description="Path with agents answers"
    )
    document_evaluations_path: str = Field(
        default="reasonings.csv",
        description="Path to write (or read) the evaluations of the retrieved documents",
    )
    answers_evaluations_path: str = Field(
        default="answers_evaluations.csv",
        description="Path to write (or read) the evaluations of each agent's answers",
    )
    games_evaluations_path: str = Field(
        default="pairwise_answer_evaluations.csv",
        description="Path to write (or read) the evaluations of the pairwise games between agents answers",
    )
    query_placeholder: str = Field(
        default="query",
        description="The placeholder for the query in the prompt",
    )
    scoring_key: str = Field(
        default="relevance",
        description="When using answer_format=json, the key to extract from the answer",
    )
    scoring_keys: List[str] = Field(
        default=["relevance"],
        description="When using answer_format=multi_field_json, the keys to extract from the answer",
    )
    answer_format: Union[str, AnswerFormat] = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM",
    )
    n_processes: int = Field(
        default=1,
        description="The number of parallel LLM calls to use for the evaluation",
    )
    output_columns: Optional[List[str]] = Field(
        default=["qid", "did", "raw_answer", "answer"],
        description="The columns to output in the CSV file",
    )


class AllConfig(BaseEvaluatorConfig):
    reasoning_path: str = Field(
        default="reasonings.csv",
        description="CSV file with the reasoning for each retrieved document",
    )
    evaluations_file: str = Field(
        default="answers_evaluations.csv",
        description="Path to write the pairwise evaluations of answers",
    )
    output_file: str = Field(
        default="agents_ranking.csv", description="Path to the output file"
    )
    retrieval_evaluator_name: str = Field(
        default="reasoner",
        description="The name of the retrieval evaluator to use",
    )
    answer_evaluator_name: str = Field(
        default="pairwise_reasoning",
        description="The name of the answer evaluator to use",
    )
    answer_ranker_name: str = Field(
        default="elo", description="The name of the answer ranker to use"
    )
    k: int = Field(default=100, description="Number of games to generate")
    initial_score: int = Field(
        default=1000, description="The initial score for each agent"
    )
    elo_k: int = Field(
        default=32, description="The K factor for the Elo ranking algorithm"
    )
    bidirectional: bool = Field(
        default=False, description="Wether or not to run each game in both directions"
    )
