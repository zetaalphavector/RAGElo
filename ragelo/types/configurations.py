from enum import StrEnum
from typing import List, Optional

from pydantic import BaseModel, Field


class AnswerFormat(StrEnum):
    """Enum that contains the names of the available answer formats"""

    JSON = "json"
    TEXT = "text"
    MULTI_FIELD_JSON = "multi_field_json"


class LLMProviderConfig(BaseModel):
    api_key: str


class OpenAIConfiguration(LLMProviderConfig):
    org: Optional[str] = None
    api_type: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    model_name: str = "gpt-3.5-turbo"


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
    model_name: str = Field(
        default="gpt-4", description="Name of the model to use for the LLM provider"
    )
    data_path: str = Field(default="data/", description="Path to the data folder")

    llm_provider: str = Field(
        default="openai", description="The name of the LLM provider"
    )
    write_output: bool = Field(
        default=True, description="Whether or not to write the output to a file"
    )
    scoring_key: Optional[list[str]] = Field(
        default="relevance",
        description="The fields to extract from the answer",
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
    answer_format: str = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM",
    )


class ReasonerEvaluatorConfig(BaseEvaluatorConfig):
    answer_format: str = Field(
        default=AnswerFormat.TEXT,
        description="The format of the answer returned by the LLM",
    )
    scoring_key: Optional[list[str]] = Field(
        default=None,
        description="The fields to extract from the answer",
    )


class DomainExpertEvaluatorConfig(BaseEvaluatorConfig):
    domain_long: str = Field(
        default="",
        description="The name of corpus domain. (e.g., Chemical Engineering)",
    )
    domain_short: Optional[str] = Field(
        default=None,
        description="A short or alternative name of the domain. "
        "(e.g., Chemistry, CS, etc.)",
    )
    company: Optional[str] = Field(
        default=None,
        description="Name of the company or organization that the user that "
        "submitted the query works for. that the domain belongs to. "
        "(e.g.: ChemCorp, CS Inc.)",
    )
    extra_guidelines: Optional[List[str]] = Field(
        default=None,
        description="A list of extra guidelines to be used when reasoning about the "
        "relevancy of the document.",
    )
    output_file: str = Field(
        default="domain_expert_evaluations.csv",
        description="Path to the output file",
    )
    answer_format: str = Field(
        default="text",
        description="The format of the answer returned by the LLM",
    )
    scoring_key: Optional[list[str]] = Field(
        default="score",
        description="The fields to extract from the answer",
    )


class CustomPromptEvaluatorConfig(BaseEvaluatorConfig):
    prompt: str = Field(
        default="query: {query} document: {document}",
        description="The prompt to be used to evaluate the documents. It should contain a {query} and a {document} placeholder",
    )
    document_placeholder: str = Field(
        default="document",
        description="The placeholder for the document in the prompt",
    )
    output_file: str = Field(
        default="custom_prompt_evaluations.csv",
        description="Path to the output file",
    )
    scoring_key: Optional[list[str]] = Field(
        default=["quality", "trustworthiness", "originality"],
        description="The fields to extract from the answer",
    )
    answer_format: str = Field(
        default=AnswerFormat.MULTI_FIELD_JSON,
        description="The format of the answer returned by the LLM",
    )


class FewShotExample(BaseModel):
    """A few-shot example."""

    passage: str
    query: str
    relevance: int
    reasoning: str


class FewShotEvaluatorConfig(BaseEvaluatorConfig):
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="The system prompt to be used to evaluate the documents.",
    )
    few_shots: List[FewShotExample] = Field(
        default=[], description="A list of few-shot examples to be used in the prompt"
    )
    few_shot_user_prompt: str = Field(
        default="Query: {query}\n\nPassage:{passage}",
        description="The individual prompt to be used to evaluate the documents. It should contain a {query} and a {passage} placeholder",
    )
    few_shot_assistant_answer: str = Field(
        default='{reasoning}\n\n{{"relevance": {relevance}}}',
        description="The expected answer format from the LLM for each evaluated document "
        "It should contain a {reasoning} and a {relevance} placeholder",
    )
    output_file: str = Field(
        default="few_shot_evaluations.csv",
        description="Path to the output file",
    )
    document_placeholder: str = Field(
        default="document",
        description="The placeholder for the document in the prompt",
    )
    reasoning_placeholder: str = Field(
        default="reasoning",
        description="The placeholder for the reasoning in the prompt",
    )
    relevance_placeholder: str = Field(
        default="relevance",
        description="The placeholder for the relevance in the prompt",
    )


class RDNAMEvaluatorConfig(BaseEvaluatorConfig):
    annotator_role: Optional[str] = Field(
        default=None,
        description="A String defining the type of user the LLM should mimic. "
        "(e.g.: 'You are a search quality rater evaluating the relevance "
        "of web pages')",
    )
    use_aspects: bool = Field(
        default=False,
        description="Should the prompt include aspects to get to the final score? "
        "If true, will prompt the LLM to compute scores for M (intent match) "
        "and T (trustworthy) for the document before computing the final score.",
    )
    use_multiple_annotators: bool = Field(
        default=False,
        description="Should the prompt ask the LLM to mimic multiple annotators?",
    )
    output_file: str = Field(
        default="rdnam_evaluations.csv", description="Path to the output file"
    )


class BaseAnswerEvaluatorConfig(BaseEvaluatorConfig):
    answers_path: str = Field(
        default="answers.csv", description="Path to the answers file"
    )
    answer_placeholder: str = Field(
        default="answer", description="The placeholder for the answer in the prompt"
    )
    documents_placeholder: str = Field(
        default="documents",
        description="The placeholder for the documents in the prompt",
    )


class PairwiseEvaluatorConfig(BaseAnswerEvaluatorConfig):
    """Configuration for the pairwise evaluator."""

    reasoning_path: str = Field(
        default="reasonings.csv",
        description="CSV file with the reasoning for each retrieved document",
    )
    reasonings: Optional[dict[str, str]] = Field(
        default=None, description="A dictionary with the reasoning for each document"
    )
    bidirectional: bool = Field(
        default=False, description="Whether or not to run each game in both directions"
    )
    k: int = Field(default=100, description="Number of games to generate")
    output_file: str = Field(
        default="pairwise_answers_evaluations.csv",
        description="Path to the output file",
    )


class CustomPromptAnswerEvaluatorConfig(BaseAnswerEvaluatorConfig):
    prompt: str = Field(
        default="retrieved documents: {documents} query: {query} answer: {answer}",
        description="The prompt to be used to evaluate the documents. It should contain a {query} and a {document} placeholder",
    )
    output_file: str = Field(
        default="custom_prompt_answers_evaluations.csv",
        description="Path to the output file",
    )
    scoring_key: Optional[list[str]] = Field(
        default=["quality", "trustworthiness", "originality"],
        description="The fields to extract from the answer",
    )
    answer_format: str = Field(
        default=AnswerFormat.MULTI_FIELD_JSON,
        description="The format of the answer returned by the LLM",
    )


class AgentRankerConfig(BaseConfig):
    evaluations_file: str = Field(
        default="data/evaluations.csv",
        description="Path with the pairwise evaluations of answers",
    )
    output_file: Optional[str] = Field(
        default=None, description="Path to the output file"
    )


class EloAgentRankerConfig(AgentRankerConfig):
    k: int = Field(default=32, description="The K factor for the Elo ranking algorithm")
    initial_score: int = Field(
        default=1000, description="The initial score for each agent"
    )
    output_file: str = Field(
        default="elo_ranking.csv", description="Path to the output file"
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
