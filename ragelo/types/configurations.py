from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMProviderConfig:
    api_key: str


@dataclass
class OpenAIConfiguration(LLMProviderConfig):
    org: Optional[str] = None
    api_type: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"


@dataclass
class BaseConfig:
    force: bool = field(
        default=False,
        metadata={
            "help": "Force the execution of the commands and overwrite any existing files"
        },
    )
    rich_print: bool = field(
        default=False, metadata={"help": "Use rich to print colorful outputs"}
    )
    verbose: bool = field(
        default=False,
        metadata={
            "help": "Wether or not to be verbose and print all intermediate steps"
        },
    )
    credentials_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a txt file with the credentials for the different LLM providers"
        },
    )
    model_name: str = field(
        default="gpt-4",
        metadata={"help": "Name of the model to use for the LLM provider"},
    )
    data_path: str = field(
        default="data/", metadata={"help": "Path to the data folder"}
    )

    llm_provider: str = field(
        default="openai",
        metadata={"help": "The name of the LLM provider"},
    )
    write_output: bool = field(
        default=True,
        metadata={"help": "Wether or not to write the output to a file"},
    )


@dataclass(kw_only=True)
class BaseEvaluatorConfig(BaseConfig):
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the output file"},
    )
    query_path: str = field(
        default="queries.csv",
        metadata={"help": "Path to the queries file"},
    )
    documents_path: str = field(
        default="documents.csv", metadata={"help": "Path to the documents file"}
    )


@dataclass(kw_only=True)
class DomainExpertEvaluatorConfig(BaseEvaluatorConfig):
    domain_long: str = field(
        default="",
        metadata={
            "help": "The name of corpus domain. "
            "(e.g., Chemical Engineering, Computer Science, etc.)"
        },
    )
    domain_short: Optional[str] = field(
        default=None,
        metadata={
            "help": "A short or alternative name of the domain. "
            "(e.g., Chemistry, CS, etc.)"
        },
    )

    company: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the company or organization that the user that "
            "submitted the query works for. that the domain belongs to. "
            "(e.g.: ChemCorp, CS Inc.)"
        },
    )
    extra_guidelines: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "A list of extra guidelines to be used when reasoning about the "
            "relevancy of the document."
        },
    )
    output_file: str = field(
        default="domain_expert_evaluations.csv",
        metadata={"help": "Path to the output file"},
    )


@dataclass(kw_only=True)
class CustomPromptEvaluatorConfig(BaseEvaluatorConfig):
    prompt: str = field(
        default="query: {query} document: {document}",
        metadata={
            "help": "The prompt to be used to evaluate the documents. It should contain a {query} and a {document} placeholder"
        },
    )
    query_placeholder: str = field(
        default="query",
        metadata={"help": "The placeholder for the query in the prompt"},
    )
    document_placeholder: str = field(
        default="document",
        metadata={"help": "The placeholder for the document in the prompt"},
    )
    output_file: str = field(
        default="custom_prompt_evaluations.csv",
        metadata={"help": "Path to the output file"},
    )


@dataclass
class FewShotExample:
    """A few-shot example."""

    passage: str
    query: str
    relevance: int
    reasoning: str


@dataclass(kw_only=True)
class FewShotEvaluatorConfig(BaseEvaluatorConfig):
    system_prompt: str = field(
        default="You are a helpful assistant.",
        metadata={"help": "The system prompt to be used to evaluate the documents."},
    )
    few_shots: List[FewShotExample] = field(
        metadata={"help": "A list of few-shot examples to be used in the prompt"}
    )
    few_shot_user_prompt: str = field(
        default="Query: {query}\n\nPassage:{passage}",
        metadata={
            "help": "The individual prompt to be used to evaluate the documents. It should contain a {query} and a {passage} placeholder"
        },
    )
    few_shot_assistant_answer: str = field(
        default='{reasoning}\n\n{{"relevance": {relevance}}}',
        metadata={
            "help": "The expected answer format from the LLM for each evaluated document "
            "It should contain a {reasoning} and a {relevance} placeholder"
        },
    )
    output_file: str = field(
        default="few_shot_evaluations.csv",
        metadata={"help": "Path to the output file"},
    )
    query_placeholder: str = field(
        default="query",
        metadata={"help": "The placeholder for the query in the prompt"},
    )
    document_placeholder: str = field(
        default="document",
        metadata={"help": "The placeholder for the document in the prompt"},
    )
    reasoning_placeholder: str = field(
        default="reasoning",
        metadata={"help": "The placeholder for the reasoning in the prompt"},
    )
    relevance_placeholder: str = field(
        default="relevance",
        metadata={"help": "The placeholder for the relevance in the prompt"},
    )


@dataclass(kw_only=True)
class RDNAMEvaluatorConfig(BaseEvaluatorConfig):
    role: Optional[str] = field(
        default=None,
        metadata={
            "help": "A String defining the type of user the LLM should mimic. "
            "(e.g.: 'You are a search quality rater evaluating the relevance "
            "of web pages')"
        },
    )
    # Configurations for the RDNAM evaluator
    aspects: bool = field(
        default=False,
        metadata={
            "help": "Should the prompt include aspects to get tot he final score? "
            "If true, will prompt the LLM to compute scores for M (intent match) "
            "and T (trustworthy) for the document before computing the final score."
        },
    )
    multiple: bool = field(
        default=False,
        metadata={
            "help": "Should the prompt ask the LLM to mimic multiple annotators?"
        },
    )
    narrative_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the file containing narratives for each query"},
    )

    description_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the file containing descriptions for each query"},
    )
    output_file: str = field(
        default="rdnam_evaluations.csv",
        metadata={"help": "Path to the output file"},
    )


@dataclass
class BaseAnswerEvaluatorConfig(BaseEvaluatorConfig):
    answers_path: str = field(
        default="answers.csv", metadata={"help": "Path to the answers file"}
    )
    reasoning_path: str = field(
        default="reasonings.csv",
        metadata={"help": "CSV file with the reasoning for each retrieved document"},
    )


@dataclass
class PairwiseEvaluatorConfig(BaseAnswerEvaluatorConfig):
    """Configuration for the pairwise evaluator."""

    bidirectional: bool = field(
        default=False,
        metadata={"help": "Wether or not to run each game in both directions"},
    )
    k: int = field(
        default=100,
        metadata={"help": "Maximum number of pairwise comparisons to generate"},
    )
    output_file: str = field(
        default="pairwise_answers_evaluations.csv",
        metadata={"help": "Path to the output file"},
    )


@dataclass
class CustomPromptAnswerEvaluatorConfig(BaseAnswerEvaluatorConfig):
    prompt: str = field(
        default="retrieved documents: {documents} query: {query} answer: {answer}",
        metadata={
            "help": "The prompt to be used to evaluate the documents. It should contain a {query} and a {document} placeholder"
        },
    )
    query_placeholder: str = field(
        default="query",
        metadata={"help": "The placeholder for the query in the prompt"},
    )
    answer_placeholder: str = field(
        default="answer",
        metadata={"help": "The placeholder for the answer in the prompt"},
    )
    documents_placeholder: str = field(
        default="documents",
        metadata={"help": "The placeholder for the documents in the prompt"},
    )
    output_file: str = field(
        default="custom_prompt_answers_evaluations.csv",
        metadata={"help": "Path to the output file"},
    )
    scoring_fields: list[str] = field(
        default_factory=lambda: ["quality", "trustworthiness", "originality"],
        metadata={"help": "The fields to extract from the answer"},
    )


@dataclass(kw_only=True)
class AgentRankerConfig(BaseConfig):
    evaluations_file: str = field(
        default="data/evaluations.csv",
        metadata={"help": "Path with the pairwise evaluations of answers"},
    )
    output_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the output file"},
    )


@dataclass(kw_only=True)
class EloAgentRankerConfig(AgentRankerConfig):
    k: int = field(
        default=32,
        metadata={"help": "The K factor for the Elo ranking algorithm"},
    )
    initial_score: int = field(
        default=1000,
        metadata={"help": "The initial score for each agent"},
    )
    output_file: str = field(
        default="elo_ranking.csv",
        metadata={"help": "Path to the output file"},
    )


@dataclass
class AllConfig(BaseEvaluatorConfig):
    reasoning_path: str = field(
        default="reasonings.csv",
        metadata={"help": "CSV file with the reasoning for each retrieved document"},
    )
    answers_path: str = field(
        default="answers.csv", metadata={"help": "Path to the answers file"}
    )
    evaluations_file: str = field(
        default="answers_evaluations.csv",
        metadata={"help": "Path to write the pairwise evaluations of answers"},
    )
    output_file: str = field(
        default="agents_ranking.csv",
        metadata={"help": "Path to the output file with the ranking of agents"},
    )
    retrieval_evaluator_name: str = field(
        default="reasoner",
        metadata={"help": "The name of the retrieval evaluator to use"},
    )
    answer_evaluator_name: str = field(
        default="pairwise_reasoning",
        metadata={"help": "The name of the answer evaluator to use"},
    )
    answer_ranker_name: str = field(
        default="elo",
        metadata={"help": "The name of the answer ranker to use"},
    )

    k: int = field(
        default=100,
        metadata={"help": "Number of games to generate"},
    )
    initial_score: int = field(
        default=1000,
        metadata={"help": "The initial score for each agent"},
    )
    elo_k: int = field(
        default=32,
        metadata={"help": "The K factor for the Elo ranking algorithm"},
    )
    bidirectional: bool = field(
        default=False,
        metadata={"help": "Wether or not to run each game in both directions"},
    )
