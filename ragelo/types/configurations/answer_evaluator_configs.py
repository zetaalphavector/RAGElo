from pydantic import Field

from ragelo.types.configurations.base_configs import BaseEvaluatorConfig
from ragelo.types.types import AnswerFormat


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

    bidirectional: bool = Field(
        default=False, description="Whether or not to run each game in both directions"
    )
    k: int = Field(default=100, description="Number of games to generate")
    output_file: str = Field(
        default="pairwise_answers_evaluations.csv",
        description="Path to the output file",
    )
    documents_path: str = Field(
        default="reasonings.csv",
        description="Path with the outputs from the reasoner Retrieval Evaluator",
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
    scoring_keys: list[str] = Field(
        default=["quality", "trustworthiness", "originality"],
        description="The fields to extract from the answer",
    )
    answer_format: str = Field(
        default=AnswerFormat.MULTI_FIELD_JSON,
        description="The format of the answer returned by the LLM",
    )
