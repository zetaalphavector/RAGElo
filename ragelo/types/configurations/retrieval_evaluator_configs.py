from typing import List, Optional

from ragelo.pydantic_v1 import Field
from ragelo.types.configurations.base_configs import AnswerFormat, BaseEvaluatorConfig
from ragelo.types.types import FewShotExample


class BaseRetrievalEvaluatorConfig(BaseEvaluatorConfig):
    document_placeholder: str = Field(
        default="document",
        description="The placeholder for the document in the prompt",
    )


class ReasonerEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    answer_format: str = Field(
        default=AnswerFormat.TEXT,
        description="The format of the answer returned by the LLM",
    )


class DomainExpertEvaluatorConfig(BaseRetrievalEvaluatorConfig):
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
    scoring_key: str = Field(
        default="score",
        description="The field to use when parsing the llm answer",
    )


class CustomPromptEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    prompt: str = Field(
        default="query: {query} document: {document}",
        description="The prompt to be used to evaluate the documents. It should contain a {query} and a {document} placeholder",
    )
    scoring_keys: list[str] = Field(
        default=["quality", "trustworthiness", "originality"],
        description="The fields to use when parsing the llm answer",
    )
    answer_format: str = Field(
        default=AnswerFormat.MULTI_FIELD_JSON,
        description="The format of the answer returned by the LLM",
    )


class FewShotEvaluatorConfig(BaseRetrievalEvaluatorConfig):
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
    reasoning_placeholder: str = Field(
        default="reasoning",
        description="The placeholder for the reasoning in the prompt",
    )
    relevance_placeholder: str = Field(
        default="relevance",
        description="The placeholder for the relevance in the prompt",
    )


class RDNAMEvaluatorConfig(BaseRetrievalEvaluatorConfig):
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
