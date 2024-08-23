from typing import List, Optional, Union

from pydantic import Field

from ragelo.types.configurations.base_configs import AnswerFormat, BaseEvaluatorConfig
from ragelo.types.types import FewShotExample, RetrievalEvaluatorTypes


class BaseRetrievalEvaluatorConfig(BaseEvaluatorConfig):
    evaluator_name: Union[
        str, RetrievalEvaluatorTypes
    ] = RetrievalEvaluatorTypes.CUSTOM_PROMPT
    document_placeholder: str = Field(
        default="document",
        description="The placeholder for the document in the prompt",
    )


class ReasonerEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    answer_format_retrieval_evaluator: Union[str, AnswerFormat] = Field(
        default="text",
        description="The format of the answer returned by the LLM",
    )
    scoring_key_retrieval_evaluator: str = Field(
        default="answer",
        description="When using answer_format=json, the key to extract from the answer",
    )
    scoring_keys_retrieval_evaluator: List[str] = Field(
        default=["relevance"],
        description="When using answer_format=multi_field_json, the keys to extract from the answer",
    )
    document_evaluations_file: str = Field(
        default="reasonings.csv",
        description="Path to write (or read) the evaluations of the retrieved documents",
    )
    output_columns_retrieval_evaluator: List[str] = Field(
        default=["qid", "did", "raw_answer", "answer"],
        description="The columns to output in the CSV file",
    )
    evaluator_name: Union[
        str, RetrievalEvaluatorTypes
    ] = RetrievalEvaluatorTypes.REASONER


class DomainExpertEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    expert_in: str = Field(
        default="",
        description="What the LLM should mimic being an expert in.",
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
    document_evaluations_file: str = Field(
        default="domain_expert_evaluations.csv",
        description="Path to write (or read) the evaluations of the retrieved documents",
    )
    answer_format_retrieval_evaluator: str = Field(
        default="text",
        description="The format of the answer returned by the LLM",
    )
    scoring_key_retrieval_evaluator: str = Field(
        default="score",
        description="The field to use when parsing the llm answer",
    )
    evaluator_name: Union[
        str, RetrievalEvaluatorTypes
    ] = RetrievalEvaluatorTypes.DOMAIN_EXPERT
    output_columns_retrieval_evaluator: List[str] = Field(
        default=["qid", "did", "reasoning", "score"],
        description="The columns to output in the CSV file",
    )


class CustomPromptEvaluatorConfig(BaseRetrievalEvaluatorConfig):
    prompt: str = Field(
        default="query: {query} document: {document}",
        description="The prompt to be used to evaluate the documents. It should contain a {query} and a {document} placeholder",
    )
    scoring_keys_retrieval_evaluator: List[str] = Field(
        default=["quality", "trustworthiness", "originality"],
        description="The fields to use when parsing the llm answer",
    )
    answer_format_retrieval_evaluator: Union[str, AnswerFormat] = Field(
        default="multi_field_json",
        description="The format of the answer returned by the LLM",
    )
    document_evaluations_file: str = Field(
        default="custom_prompt_evaluations.csv",
        description="Path to write (or read) the evaluations of the retrieved documents",
    )
    evaluator_name: Union[
        str, RetrievalEvaluatorTypes
    ] = RetrievalEvaluatorTypes.CUSTOM_PROMPT


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
    document_evaluations_file: str = Field(
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
    evaluator_name: Union[
        str, RetrievalEvaluatorTypes
    ] = RetrievalEvaluatorTypes.FEW_SHOT


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
    document_evaluations_file: str = Field(
        default="rdnam_evaluations.csv", description="Path to the output file"
    )
    scoring_key_retrieval_evaluator: str = Field(
        default="answer",
        description="The field to use when parsing the llm answer",
    )
    evaluator_name: Union[str, RetrievalEvaluatorTypes] = RetrievalEvaluatorTypes.RDNAM
