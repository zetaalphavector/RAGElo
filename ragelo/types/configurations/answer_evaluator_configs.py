from typing import Optional

from pydantic import Field

from ragelo.types.configurations.base_configs import AnswerFormat, BaseEvaluatorConfig


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
    pairwise: bool = Field(
        default=False, description="Whether or not to the evaluator is pairwise"
    )
    output_columns: list[str] = Field(
        default=["qid", "agent", "raw_answer", "answer"],
        description="The columns to output in the CSV file",
    )
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


class PairwiseEvaluatorConfig(BaseAnswerEvaluatorConfig):
    """Configuration for the pairwise evaluator."""

    bidirectional: bool = Field(
        default=False, description="Whether or not to run each game in both directions"
    )
    k: int = Field(default=100, description="Number of games to generate")
    games_evaluations_path: str = Field(
        default="pairwise_answers_evaluations.csv",
        description="Path to the output file",
    )
    documents_path: str = Field(
        default="reasonings.csv",
        description="Path with the outputs from the reasoner Retrieval Evaluator",
    )
    pairwise: bool = Field(
        default=True, description="Whether or not to the evaluator is pairwise"
    )
    document_template: str = Field(
        default="[{did}] {annotation}",
        description="The template to format each individual document in the prompt",
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Prompt to use for the evaluator. If not provided, a default prompt will be used",
    )
    factors: str = Field(
        default=(
            "the correctness, helpfulness, completeness, accuracy, depth, and "
            "level of detail of their responses"
        ),
        description=(
            "A string containing the factors to be used when evaluating an answer. "
            "If not provided, a default string will be used"
        ),
    )


class CustomPromptAnswerEvaluatorConfig(BaseAnswerEvaluatorConfig):
    prompt: str = Field(
        default="retrieved documents: {documents} query: {query} answer: {answer}",
        description="The prompt to be used to evaluate the documents. It should contain a {query} and a {document} placeholder",
    )
    answers_evaluations_path: str = Field(
        default="custom_prompt_answers_evaluations.csv",
        description="Path to the output file",
    )
    scoring_keys: list[str] = Field(
        default=["quality", "trustworthiness", "originality"],
        description="The fields to extract from the answer",
    )
    answer_format: str = Field(
        default=AnswerFormat.MULTI_FIELD_JSON,
        description="The format of the answer returned by the LLM",
    )
