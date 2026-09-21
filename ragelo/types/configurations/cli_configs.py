from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from jinja2 import Template
from pydantic import AliasChoices, Field

from ragelo.types.configurations.answer_evaluator_configs import (
    PairwiseDomainExpertEvaluatorConfig,
    PairwiseEvaluatorConfig,
)
from ragelo.types.configurations.base_configs import BaseConfig
from ragelo.types.configurations.benchmark_configs import BenchmarkDatasetConfig, TrecRag24AnswersDatasetConfig
from ragelo.types.configurations.retrieval_evaluator_configs import (
    DomainExpertEvaluatorConfig,
    RDNAMEvaluatorConfig,
    ReasonerEvaluatorConfig,
)

Positional = Annotated[str, typer.Argument()]


class BaseCLIConfig(BaseConfig):
    experiment_name: str = Field(
        default="experiment",
        description="The name of the experiment to run. This is also used as the local cache file name",
    )
    data_dir: str = Field(
        default="data",
        description="The directory where the data is stored.",
    )
    queries_csv_file: Positional = Field(
        default="queries.csv",
        description="The path to the queries CSV file. The file should contain at least the following columns: "
        "qid, query. Any additional columns will be considered as metadata.",
    )
    show_results: bool = Field(
        default=True,
        description="Whether to render evaluation result tables and summaries to the console.",
        validation_alias=AliasChoices("show_results", "verbose"),
    )
    output_file: str | None = Field(
        default=None,
        description="The path to the output file where the results will be saved.",
    )
    rich_print: bool = Field(
        default=True,
        description="Use rich to print colorful outputs.",
    )
    model: str | None = Field(
        default=None,
        description="The model to use for the LLM. The openai provider has a default. The vercel and ollama "
        "providers need one, such as anthropic/claude-haiku-4-5 or llama3.1.",
    )


class CLIEvaluatorConfig(BaseCLIConfig):
    documents_csv_file: Positional = Field(
        default="documents.csv",
        description=(
            "The path to the documents CSV file. The file should contain at least the following columns: "
            "qid, did, document. Any additional columns will be considered as metadata."
        ),
    )
    answers_csv_file: Positional = Field(
        default="answers.csv",
        description="The path to the answers CSV file. The file should contain at least the following columns: "
        "qid, agent, answer. Any additional columns will be considered as metadata. Ignored on Retrieval Evaluators.",
    )


class CLIDomainExpertEvaluatorConfig(CLIEvaluatorConfig, DomainExpertEvaluatorConfig):
    expert_in: str = " "
    user_prompt: Template | None = None


class CLIReasonerEvaluatorConfig(CLIEvaluatorConfig, ReasonerEvaluatorConfig):
    user_prompt: Template | None = None


class CLIRDNAMEvaluatorConfig(CLIEvaluatorConfig, RDNAMEvaluatorConfig):
    user_prompt: Template | None = None


class CLIPairwiseDomainExpertEvaluatorConfig(CLIEvaluatorConfig, PairwiseDomainExpertEvaluatorConfig):
    add_reasoning: bool = Field(
        default=False,
        description="If set to True, a reasoning retrieval evaluator will run, and the reasoning of the quality "
        "  of the retrieved results will be included in the prompt for the pairwise games.",
    )
    expert_in: str = " "


class CLIPairwiseEvaluatorConfig(CLIEvaluatorConfig, PairwiseEvaluatorConfig):
    add_reasoning: bool = Field(
        default=False,
        description="If set to True, a reasoning retrieval evaluator will run, and the reasoning of the quality "
        "  of the retrieved results will be included in the prompt for the pairwise games.",
    )


class CLIConfig(BaseCLIConfig):
    documents_csv_file: Positional = Field(
        default="documents.csv",
        description="The path to the documents CSV file. The file should contain at least the following columns: "
        "qid, did, document. Any additional columns will be considered as metadata.",
    )
    answers_csv_file: Positional = Field(
        default="answers.csv",
        description="The path to the answers CSV file. The file should contain at least the following columns: "
        "qid, did, answer. Any additional columns will be considered as metadata.",
    )

    k: int = Field(default=100, description="Number of pairwise games to generate")
    initial_score: int = Field(default=1000, description="The initial Elo score for each agent")
    elo_k: int = Field(default=32, description="The K factor for the Elo ranking algorithm")


class CLIBenchmarkConfig(BenchmarkDatasetConfig):
    models: list[str] | None = Field(default=None, description="The models to judge with.")
    evaluators: list[str] | None = Field(
        default=None,
        description="The evaluators to judge with. A retrieval dataset takes the names in "
        "ragelo/benchmarks/variants.py, the others take the name of a pairwise answer evaluator.",
    )
    prices: list[str] | None = Field(
        default=None,
        description="MODEL=INPUT,OUTPUT,CACHED in USD per million tokens. Without it only the tokens are reported.",
    )
    output_dir: Path | None = Field(
        default=None,
        description="The directory where the judgments are cached. Defaults to benchmarks/results/<dataset>.",
    )
    tag: str = Field(default="", description="Suffix of the experiment names, to judge again beside an old cache.")
    n_processes: int = Field(default=16, description="The number of parallel LLM calls to use for the evaluation.")


class CLITrecRag24AnswersBenchmarkConfig(CLIBenchmarkConfig, TrecRag24AnswersDatasetConfig):
    pass
