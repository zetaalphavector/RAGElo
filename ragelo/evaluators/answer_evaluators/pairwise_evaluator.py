from __future__ import annotations

from typing import Literal

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
    BaseAnswerEvaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
from ragelo.types.configurations import PairwiseEvaluatorConfig
from ragelo.types.evaluables import PairwiseGame
from ragelo.types.formats import AnswerFormat, LLMResponseType
from ragelo.types.query import Query
from ragelo.types.types import AnswerEvaluatorTypes


class PairWiseAnswerAnswerFormat(PydanticBaseModel):
    winner: Literal["A", "B", "C"] = Field(..., description="The winner of the pairwise comparison.")


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.PAIRWISE)
class PairwiseAnswerEvaluator(BaseAnswerEvaluator):
    """An evaluator that evaluates RAG-based answers pairwise, with document reasoning and citations."""

    config: PairwiseEvaluatorConfig
    citations_prompt = " Answers cite documents using square brackets."
    document_template_raw_only = "[{did}] {doc}"
    document_template_annotation_only = "[{did}] {annotation}"
    document_template_raw_and_annotation = "[RETRIEVED DOCUMENT]\n{doc}\n[DOCUMENT RELEVANCE]\n{annotation}\n"
    documents_prompt_relevance_only = (
        "For each reference document, you will be provided with a reasoning "
        "explaining why the document is or is not relevant."
    )
    documents_prompt_raw_and_relevance = (
        "For each reference document, you will be provided with the text "
        "of the document as well as a reasoning  why the document "
        "is or is not relevant."
    )
    documents_prompt_raw_only = "You will be provided with the text of each reference document."

    prompt = """
Please act as an impartial judge and evaluate the quality of the responses provided \
by two AI assistants tasked to answer the question displayed below, based on a set \
of documents retrieved by a search engine.
You should choose the assistant that best answers the user question based on a set \
of reference documents that may or not be relevant.{citations}
{document_rel}
Your evaluation should consider factors such as {factors}.
Details are only useful if they answer the user question. If an answer \
contains non-relevant details, it should not be preferred over one that only \
use relevant information.
Begin your evaluation by explaining whether or nor each answer correctly answers the user \
question. Then, you should compare the two responses and provide a short explanation \
on their differences. Avoid any position biases and ensure that the order in which \
the responses were presented does not influence your decision. Do not allow the \
length of the responses to influence your evaluation. Be as objective as possible.
After providing your explanation, output your final verdict by strictly following \
this format: "A" if assistant A is better, "B" if assistant B is better, \
and "C" for a tie.

[User Question]
{query}

[Reference Documents]
{documents}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
""".strip()

    def __init__(
        self,
        config: PairwiseEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        self.factors = config.factors
        if config.include_annotations and config.include_raw_documents:
            config.document_template = self.document_template_raw_and_annotation
            self.documents_prompt = self.documents_prompt_raw_and_relevance
        elif config.include_annotations:
            config.document_template = self.document_template_annotation_only
            self.documents_prompt = self.documents_prompt_relevance_only
        elif config.include_raw_documents:
            config.document_template = self.document_template_raw_only
            self.documents_prompt = self.documents_prompt_raw_only
        if config.prompt:
            self.prompt = config.prompt
        if config.llm_answer_format == AnswerFormat.STRUCTURED:
            self.config.llm_response_schema = PairWiseAnswerAnswerFormat
        elif self.config.llm_answer_format != AnswerFormat.JSON:
            logger.warning("We are using a PairwiseAnswerEvaluator config. Forcing the LLM answer format to JSON.")
            self.config.llm_answer_format = AnswerFormat.JSON
            self.config.llm_response_schema = self.config.llm_response_schema or {
                "analysis_assistant_a": "A string with your analysis of assistant A's answer",
                "analysis_assistant_b": "A string with your analysis of assistant B's answer",
                "differences": "A string with your comparison between the two answers and their differences",
                "winner": (
                    "The winner of the comparison. "
                    "'A' if assistant A is better, 'B' if assistant B is better, and 'C' for a tie"
                ),
            }

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> str | list[dict[str, str]]:
        documents = self._prepare_documents(query)

        query_metadata = self._get_usable_fields_from_metadata(
            self.prompt, query.metadata, skip_fields=[self.config.query_placeholder]
        )
        answer_a_metadata = self._get_usable_fields_from_metadata(
            self.prompt,
            game.agent_a_answer.metadata,
            skip_fields=[self.config.answer_placeholder],
        )
        answer_b_metadata = self._get_usable_fields_from_metadata(
            self.prompt,
            game.agent_b_answer.metadata,
            skip_fields=[self.config.answer_placeholder],
        )
        if self.config.has_citations:
            citations = self.citations_prompt
        else:
            citations = ""
        formatters = {
            self.config.query_placeholder: query.query,
            self.config.documents_placeholder: documents,
            "answer_a": game.agent_a_answer.text,
            "answer_b": game.agent_b_answer.text,
            "citations": citations,
            "factors": self.factors,
            "document_rel": self.documents_prompt,
            **query_metadata,
            **answer_a_metadata,
            **answer_b_metadata,
        }
        return self.prompt.format(**formatters)

    def _process_answer(self, llm_response: LLMResponseType) -> LLMResponseType:
        """Extracts the relevant part of an answer."""
        if isinstance(llm_response.parsed_answer, dict):
            answer = llm_response.parsed_answer["winner"]
        elif isinstance(llm_response.parsed_answer, PairWiseAnswerAnswerFormat):
            answer = llm_response.parsed_answer.winner
        else:
            answer = llm_response.parsed_answer
        return LLMResponseType(
            raw_answer=llm_response.raw_answer,
            parsed_answer=answer,
        )
