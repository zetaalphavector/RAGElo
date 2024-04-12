from typing import Optional

from pydantic import BaseModel, Field

from ragelo.types.types import AnswerFormat


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
    model: str = Field(
        default="gpt-4-turbo",
        description="Name of the model to use for the LLM provider",
    )
    data_path: str = Field(default="data/", description="Path to the data folder")

    llm_provider: str = Field(
        default="openai", description="The name of the LLM provider"
    )
    write_output: bool = Field(
        default=True, description="Whether or not to write the output to a file"
    )


class BaseEvaluatorConfig(BaseConfig):
    query_path: str = Field(
        default="queries.csv", description="Path to the queries file"
    )
    documents_path: str = Field(
        default="documents.csv", description="Path to the documents file"
    )
    output_file: Optional[str] = Field(
        default=None, description="Path to the output file"
    )
    query_placeholder: str = Field(
        default="query",
        description="The placeholder for the query in the prompt",
    )
    scoring_key: str = Field(
        default="relevance",
        description="When using answer_format=json, the key to extract from the answer",
    )
    scoring_keys: list[str] = Field(
        default=["relevance"],
        description="When using answer_format=multi_field_json, the keys to extract from the answer",
    )
    answer_format: str = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM",
    )


class AllConfig(BaseEvaluatorConfig):
    reasoning_path: str = Field(
        default="reasonings.csv",
        description="CSV file with the reasoning for each retrieved document",
    )
    answers_path: str = Field(
        default="answers.csv", description="Path to the answers file"
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
