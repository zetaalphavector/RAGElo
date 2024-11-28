from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from ragelo.logger import logger
from ragelo.types.configurations.base_configs import AnswerFormat, BaseEvaluatorConfig
from ragelo.types.pydantic_models import BaseModel, post_validator
from ragelo.types.types import RetrievalEvaluatorTypes


class FewShotExample(BaseModel):
    """A few-shot example used in the few-shot retrieval evaluator.
    Args:
        passage str: The passage of the example.
        query str: The query of the example.
        relevance int: The relevance of the example.
        reasoning str: The reasoning behind the relevance
    """

    passage: str
    query: str
    relevance: int
    reasoning: str


class BaseRetrievalEvaluatorConfig(BaseEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.CUSTOM_PROMPT
    document_placeholder: str = Field(
        default="document",
        description="The placeholder for the document in the prompt",
    )


class ReasonerEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.REASONER
    llm_answer_format: AnswerFormat = Field(
        default=AnswerFormat.TEXT,
        description="The format of the answer returned by the LLM",
    )

    @post_validator
    @classmethod
    def check_answer_format(cls, values):
        if values.llm_answer_format != AnswerFormat.TEXT:
            logger.warning("We are using the Reasoner Evaluator config. Forcing the LLM answer format to TEXT.")
            values.llm_answer_format = AnswerFormat.TEXT
        return values


class DomainExpertEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.DOMAIN_EXPERT
    expert_in: str = Field(description="What the LLM should mimic being an expert in.")
    domain_short: str | None = Field(
        default=None,
        description="A short or alternative name of the domain. " "(e.g., Chemistry, CS, etc.)",
    )
    company: str | None = Field(
        default=None,
        description="Name of the company or organization that the user that "
        "submitted the query works for. that the domain belongs to. "
        "(e.g.: ChemCorp, CS Inc.)",
    )
    extra_guidelines: list[str] | None = Field(
        default=None,
        description="A list of extra guidelines to be used when reasoning about the " "relevancy of the document.",
    )
    llm_answer_format: AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM",
    )
    llm_response_schema: Type[PydanticBaseModel] | dict[str, Any] | None = Field(
        default={
            "score": (
                "An integer between 0 and 2 representing the score of the document, "
                "where 0 means the document is not relevant to the query, 1 means the document is somewhat relevant, "
                "and 2 means the document is highly relevant."
            )
        },
    )


class CustomPromptEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.CUSTOM_PROMPT
    prompt: str = Field(
        default="query: {query} document: {document}",
        description=(
            "The prompt to be used to evaluate the documents. "
            "It should contain a {query} and a {document} placeholder"
        ),
    )


class FewShotEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.FEW_SHOT
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="The system prompt to be used to evaluate the documents.",
    )
    few_shots: list[FewShotExample] = Field(
        default_factory=list,
        description="A list of few-shot examples to be used in the prompt",
    )
    few_shot_user_prompt: str = Field(
        default="Query: {query}\n\nPassage:{passage}",
        description=(
            "The individual prompt to be used to evaluate the documents. "
            "It should contain a {query} and a {passage} placeholder"
        ),
    )
    few_shot_assistant_answer: str = Field(
        default='{reasoning}\n\n{{"relevance": {relevance}}}',
        description="The expected answer format from the LLM for each evaluated document "
        "It should contain a {reasoning} and a {relevance} placeholder",
    )
    reasoning_placeholder: str = Field(
        default="reasoning",
        description="The placeholder for the reasoning in the prompt",
    )
    relevance_placeholder: str = Field(
        default="relevance",
        description="The placeholder for the relevance in the prompt",
    )


class RDNAMEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    evaluator_name: str | RetrievalEvaluatorTypes = RetrievalEvaluatorTypes.RDNAM
    annotator_role: str | None = Field(
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
    llm_answer_format: AnswerFormat = Field(
        default=AnswerFormat.JSON,
        description="The format of the answer returned by the LLM",
    )

    @post_validator
    @classmethod
    def check_answer_format(cls, values):
        if values.llm_answer_format != AnswerFormat.JSON:
            logger.warning("We are using the RDNAM Evaluator config. Forcing the LLM answer format to JSON.")
            values.llm_answer_format = AnswerFormat.JSON
        return values
