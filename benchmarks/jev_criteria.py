"""A relevance scale asked of Jev as one yes/no criterion per grade boundary, all in a single request."""

from __future__ import annotations

from pydantic import BaseModel

from ragelo.evaluators.jev_evaluator_mixin import JEV_REASONING
from ragelo.evaluators.retrieval_evaluators.jev_evaluator import JevRubricCoverageEvaluator
from ragelo.types import Document, LLMInputPrompt, LLMResponseType, Query
from ragelo.types.answer_formats import Criterion, CriterionEvaluationPointwise, RubricCoverageAnswerFormat

# The grading rule of the RDNAM prompt (Thomas et al., arXiv 2309.10621), one question per grade boundary.
RDNAM_CRITERIA = [
    Criterion(
        criterion_name="usable",
        short_question="Assume you are writing a report on the topic of the user question. "
        "Would you use any of the information contained in the document in that report?",
    ),
    Criterion(
        criterion_name="vital",
        short_question="Is the document primarily about the topic of the user question, "
        "or does it contain vital information about that topic?",
    ),
]

# The TREC Deep Learning grades the LLMJudge assessors used, one question per grade boundary.
TREC_CRITERIA = [
    Criterion(criterion_name="related", short_question="Is the passage related to the user question?"),
    Criterion(
        criterion_name="answers",
        short_question="Does the passage have some answer for the user question, "
        "even if it is a bit unclear or hidden amongst extraneous information?",
    ),
    Criterion(
        criterion_name="exact",
        short_question="Is the passage dedicated to the user question and does it contain the exact answer?",
    ),
]


class JevCriteriaEvaluator(JevRubricCoverageEvaluator):
    """Asks each criterion as written and keeps Jev's probabilities.

    With one criterion per grade boundary, the probabilities of a yes add up to the expected grade:
    E[grade] = P(grade >= 1) + P(grade >= 2) + ...
    """

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        prompt = super()._build_message(query, document)
        questions = {
            criterion.criterion_name: {"type": "boolean", "instructions": criterion.short_question}
            for criterion in query.rubric
        }
        return prompt.model_copy(update={"questions": questions})

    def _process_answer(self, llm_response: LLMResponseType[BaseModel], query: Query) -> LLMResponseType[BaseModel]:
        criteria = [
            CriterionEvaluationPointwise(
                criterion=criterion,
                reasoning="",
                fulfillment=self._jev_answer(llm_response, criterion.criterion_name).probability or 0.0,
            )
            for criterion in query.rubric
        ]
        answer = RubricCoverageAnswerFormat(
            reasoning=JEV_REASONING,
            criteria_addressed=[item.criterion.criterion_name for item in criteria if item.fulfillment >= 0.5],
            criteria=criteria,
            score=sum(float(item.fulfillment) for item in criteria),
            rubric_fingerprint=query.rubric_fingerprint,
        )
        return LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=answer)
