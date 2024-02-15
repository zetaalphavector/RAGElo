from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMProviderConfiguration:
    api_key: str


@dataclass
class OpenAIConfiguration(LLMProviderConfiguration):
    api_key: str
    openai_org: Optional[str] = None
    openai_api_type: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_api_version: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"


@dataclass
class EvaluatorConfig:
    query_path: str = field(
        default="queries.csv", metadata={"help": "Path to the queries file"}
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to write the output of the evaluator. If not provided, will use a .log file locally."
        },
    )
    verbose: bool = False
    force: bool = False
    rich_print: bool = False


@dataclass
class RetrievalEvaluatorConfig(EvaluatorConfig):
    documents_path: str = field(
        default="documents.csv", metadata={"help": "Path to the documents file"}
    )
    # Configurations for the domain_expert evaluator
    domain_long: Optional[str] = None
    domain_short: Optional[str] = None
    company: Optional[str] = None
    extra_guidelines: Optional[str] = None

    # Configurations for the RDNAM evaluator
    role: Optional[str] = None
    aspects: bool = False
    multiple: bool = False
    narrative_file: Optional[str] = None
    description_file: Optional[str] = None


@dataclass
class AnswerEvaluatorConfig(EvaluatorConfig):
    answers_file: Optional[str] = field(
        default="answers.csv", metadata={"help": "Path to the answers file"}
    )
    # Configurations for the pairwise evaluator
    reasoning_file: Optional[str] = None
    bidirectional: bool = False
    k: int = 100
