from __future__ import annotations

from typing import Any, Callable, Type

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from ragelo.types.configurations.base_configs import BaseEvaluatorConfig
from ragelo.types.formats import AnswerFormat
from ragelo.types.types import AnswerEvaluatorTypes


class BaseAnswerEvaluatorConfig(BaseEvaluatorConfig):
    evaluator_name: str | AnswerEvaluatorTypes = ""
    answer_placeholder: str = Field(default="answer", description="The placeholder for the answer in the prompt")
    documents_placeholder: str = Field(
        default="documents",
        description="The placeholder for the documents in the prompt",
    )
    pairwise: bool = Field(default=False, description="Whether or not to the evaluator is pairwise")
    document_template: str = Field(
        default="[{did}] {doc}",
        description="The template to format each individual document in the prompt",
    )
    has_citations: bool = Field(
        default=True,
        description=(
            "Whether or not the answers contain document citations in square brackets. "
            "If used, the document_ids in the documents and in the answers should match."
        ),
    )
    include_annotations: bool = Field(
        default=True,
        description="Whether or not to include the document relevance annotations in the prompt",
    )
    include_raw_documents: bool = Field(
        default=False,
        description="Whether or not to include the raw documents in the prompt",
    )
    document_filter: Callable[[str], bool] | None = Field(
        default=None, description="A function to filter the documents"
    )
    document_relevance_threshold: int | None = Field(
        default=None,
        description="The minimum relevance score for a document to be included in the prompt. By default, all documents are included.",
    )


class PairwiseEvaluatorConfig(BaseAnswerEvaluatorConfig):
    """Configuration for the pairwise evaluator."""

    evaluator_name: str | AnswerEvaluatorTypes = AnswerEvaluatorTypes.PAIRWISE
    bidirectional: bool = Field(default=False, description="Whether or not to run each game in both directions")
    n_games_per_query: int = Field(default=100, description="Maximum number of games to generate for each query")
    games_evaluations_path: str = Field(
        default="pairwise_answers_evaluations.csv",
        description="Path to the output file",
    )
    pairwise: bool = Field(default=True, description="Whether or not to the evaluator is pairwise")
    document_template: str = Field(
        default="[{did}] {annotation}",
        description="The template to format each individual document in the prompt",
    )
    prompt: str | None = Field(
        default=None,
        description="Prompt to use for the evaluator. If not provided, a default prompt will be used",
    )
    factors: str = Field(
        default=(
            "the correctness, helpfulness, completeness, accuracy, depth, and " "level of detail of their responses"
        ),
        description=(
            "A string containing the factors to be used when evaluating an answer. "
            "If not provided, a default string will be used"
        ),
    )
    llm_answer_format: str | AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM",
    )
    llm_response_schema: Type[PydanticBaseModel] | dict[str, Any] | None = Field(
        default={
            "analysis_assistant_a": "A string with your analysis of assistant A's answer",
            "analysis_assistant_b": "A string with your analysis of assistant B's answer",
            "differences": "A string with your comparison between the two answers and their differences",
            "winner": "The winner of the comparison. 'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie",
        },
        description="The response schema for the LLM. Required if the llm_answer_format is structured and recommended for JSON.",
    )


class CustomPairwiseEvaluatorConfig(BaseAnswerEvaluatorConfig):
    """Configuration for a custom pairwise evaluator."""

    evaluator_name: str | AnswerEvaluatorTypes = AnswerEvaluatorTypes.CUSTOM_PAIRWISE
    system_prompt: str = Field(default=None, description="System prompt to use for the evaluator")
    user_prompt: str = Field(default=None, description="User prompt to use for the evaluator.")
    bidirectional: bool = Field(default=False, description="Whether or not to run each game in both directions")
    n_games_per_query: int = Field(default=100, description="Maximum number of games to generate for each query")
    pairwise: bool = Field(default=True, description="Whether or not to the evaluator is pairwise")
    llm_answer_format: str | AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM",
    )
    llm_response_schema: Type[PydanticBaseModel] | dict[str, Any] | None = Field(
        default={
            "analysis_assistant_a": "A string with your analysis of assistant A's answer",
            "analysis_assistant_b": "A string with your analysis of assistant B's answer",
            "differences": "A string with your comparison between the two answers and their differences",
            "winner": "The winner of the comparison. 'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie",
        },
        description="The response schema for the LLM. Required if the llm_answer_format is structured and recommended for JSON.",
    )


class CustomPromptAnswerEvaluatorConfig(BaseAnswerEvaluatorConfig):
    evaluator_name: str | AnswerEvaluatorTypes = AnswerEvaluatorTypes.CUSTOM_PROMPT
    prompt: str = Field(
        default="retrieved documents: {documents} query: {query} answer: {answer}",
        description="The prompt to be used to evaluate the documents. It should contain a {query} and a {document} placeholder",
    )
    answers_evaluations_path: str = Field(
        default="custom_prompt_answers_evaluations.csv",
        description="Path to the output file",
    )
    scoring_keys_answer_evaluator: list[str] = Field(
        default=["quality", "trustworthiness", "originality"],
        description="The fields to extract from the answer",
    )
    llm_answer_format: str | AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM",
    )


class PairwiseDomainExpertEvaluatorConfig(PairwiseEvaluatorConfig):
    evaluator_name: str | AnswerEvaluatorTypes = AnswerEvaluatorTypes.DOMAIN_EXPERT
    expert_in: str = Field(
        default="",
        description="What the LLM should mimic being an expert in.",
    )
    company: str | None = Field(
        default=None,
        description="Name of the company or organization that the user that "
        "submitted the query works for. that the domain belongs to. "
        "(e.g.: ChemCorp, CS Inc.)",
    )
