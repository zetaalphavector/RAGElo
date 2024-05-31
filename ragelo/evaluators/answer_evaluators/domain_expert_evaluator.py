"""Answer Evaluator with a domain expert persona"""

from typing import Dict, List, Union

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
)
from ragelo.evaluators.answer_evaluators.pairwise_evaluator import (
    PairwiseAnswerEvaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import AnswerEvaluatorTypes, PairwiseGame, Query
from ragelo.types.configurations import PairwiseDomainExpertEvaluatorConfig


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.DOMAIN_EXPERT)
class PairwiseDomainExpertEvaluator(PairwiseAnswerEvaluator):
    config: PairwiseDomainExpertEvaluatorConfig
    prompt = """
{company_prompt}You are a domain expert in {expert_in}. Your task is to \
evaluate the quality of the responses provided by two AI assistants \
tasked to answer the question shown below, based on a set \
of documents retrieved by a search engine.
You should choose the assistant that best answers the user question based on a set \
of reference documents that may or not be relevant.{citations}
{document_rel}
Your evaluation should consider factors such as {factors}.
Details are only useful if they answer the user question. If an answer \
contains non-relevant details, it should not be preferred over one that only \
use relevant information.
Begin your evaluation by explaining why each answer correctly answers the user \
question. Then, you should compare the two responses and provide a short explanation \
on their differences. Avoid any position biases and ensure that the order in which \
the responses were presented does not influence your decision. Do not allow the \
length of the responses to influence your evaluation. Be as objective as possible.
After providing your explanation, output your final verdict by strictly following \
this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, \
and "[[C]]" for a tie.

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
    COMPANY_PROMPT = "You work for {company}. "

    def __init__(
        self,
        config: PairwiseDomainExpertEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        if not self.config.expert_in:
            raise ValueError(
                "You are trying to use the Domain Expert Answer Evaluator. "
                "For this evaluator, you need to provide the domain the evaluator "
                "is an expert in the expert_in field."
            )
        self.expert_in = self.config.expert_in
        self.company = self.config.company

    def _build_message_pairwise(
        self, query: Query, game: PairwiseGame
    ) -> Union[str, List[Dict[str, str]]]:
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
            "expert_in": self.expert_in,
            **query_metadata,
            **answer_a_metadata,
            **answer_b_metadata,
        }
        if self.company:
            formatters["company_prompt"] = self.COMPANY_PROMPT.format(
                company=self.company
            )
        return self.prompt.format(**formatters)
