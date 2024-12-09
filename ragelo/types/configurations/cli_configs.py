from __future__ import annotations

from pydantic import Field

from ragelo.types.configurations.answer_evaluator_configs import (
    PairwiseDomainExpertEvaluatorConfig,
    PairwiseEvaluatorConfig,
)
from ragelo.types.configurations.base_configs import BaseConfig
from ragelo.types.configurations.retrieval_evaluator_configs import (
    DomainExpertEvaluatorConfig,
    RDNAMEvaluatorConfig,
    ReasonerEvaluatorConfig,
)


class BaseCLIConfig(BaseConfig):
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
    output_file: str | None = Field(
        default=None,
        description="The path to the output file where the results will be saved.",
    )
    save_results: bool = Field(
        default=True,
        description="Whether or not to save the results to disk.",
    )
    rich_print: bool = Field(
        default=True,
        description="Use rich to print colorful outputs.",
    )


class CLIEvaluatorConfig(BaseCLIConfig):
    documents_csv_file: str = Field(
        default="documents.csv",
        description=(
            "The path to the documents CSV file. The file should contain at least the following columns: "
            "qid, did, document. Any additional columns will be considered as metadata."
        ),
    )
    answers_csv_file: str = Field(
        default="answers.csv",
        description="The path to the answers CSV file. The file should contain at least the following columns: "
        "qid, agent, answer. Any additional columns will be considered as metadata. Ignored on Retrieval Evaluators.",
    )


class CLIDomainExpertEvaluatorConfig(CLIEvaluatorConfig, DomainExpertEvaluatorConfig):
    expert_in: str = " "


class CLIReasonerEvaluatorConfig(CLIEvaluatorConfig, ReasonerEvaluatorConfig):
    pass


class CLIRDNAMEvaluatorConfig(CLIEvaluatorConfig, RDNAMEvaluatorConfig):
    pass


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

    k: int = Field(default=100, description="Number of pairwise games to generate")
    initial_score: int = Field(default=1000, description="The initial Elo score for each agent")
    elo_k: int = Field(default=32, description="The K factor for the Elo ranking algorithm")
    bidirectional: bool = Field(default=False, description="Wether or not to run each game in both directions")
    model: str = Field(default="gpt-4o-mini", description="The model to use for the LLM")
