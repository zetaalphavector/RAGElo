import logging

from pydantic import BaseModel

from ragelo.evaluators.jev_evaluator_mixin import JEV_REASONING, JevEvaluatorMixin, JevRubricEvaluatorMixin
from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.evaluators.retrieval_evaluators.reasoner_evaluator import ReasonerEvaluator
from ragelo.evaluators.retrieval_evaluators.rubric_coverage_evaluator import RubricCoverageEvaluator
from ragelo.types import Document, LLMInputPrompt, LLMResponseType, Query
from ragelo.types.answer_formats import RetrievalEvaluationAnswer
from ragelo.types.configurations import JevRetrievalEvaluatorConfig, JevRubricCoverageEvaluatorConfig
from ragelo.types.formats import JevAnswer
from ragelo.types.types import RetrievalEvaluatorTypes
from ragelo.utils import string_to_template

logger = logging.getLogger(__name__)


def _trimmed(document: Document, max_chars: int) -> Document:
    if len(document.text) <= max_chars:
        return document
    logger.warning(f"Document {document.did} cut from {len(document.text)} to {max_chars} characters for Jev")
    return document.model_copy(update={"text": document.text[:max_chars]})


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.JEV)
class JevRetrievalEvaluator(JevEvaluatorMixin, BaseRetrievalEvaluator[JevRetrievalEvaluatorConfig]):
    """Asks Jev whether the document helps answer the question. On LLMJudge this agrees with the human
    labels better than a score question over the relevance grades, which `boolean_question=False` asks.
    """

    config: JevRetrievalEvaluatorConfig
    relevance_grades = ReasonerEvaluator.relevance_grades
    system_prompt = string_to_template("How relevant is the document to the user question?")
    user_prompt = ReasonerEvaluator.user_prompt

    boolean_prompt = "Does the document contain information that helps answer the user question?"

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        prompt = super()._build_message(query, _trimmed(document, self.config.max_document_chars))
        if self.config.boolean_question:
            return prompt.model_copy(
                update={"system_prompt": self.boolean_prompt, "questions": {"score": {"type": "boolean"}}}
            )
        question = {"type": "score", "criteria": list(self.relevance_grades)}
        return prompt.model_copy(update={"questions": {"score": question}})

    def _process_answer(self, llm_response: LLMResponseType[BaseModel], query: Query) -> LLMResponseType[BaseModel]:
        answer = self._jev_answer(llm_response, "score")
        parsed = RetrievalEvaluationAnswer(
            reasoning=JEV_REASONING,
            score=self._score(answer),
            probabilities=answer.probabilities,
            confidence=answer.confidence,
        )
        return LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=parsed)

    def _score(self, answer: JevAnswer) -> float:
        if answer.probability is not None:
            return answer.probability * self.max_score
        return int(answer.label)


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.JEV_RUBRIC_COVERAGE)
class JevRubricCoverageEvaluator(JevRubricEvaluatorMixin, RubricCoverageEvaluator):
    config: JevRubricCoverageEvaluatorConfig

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        questions = {
            criterion.criterion_name: {
                "type": "boolean",
                "instructions": "Does the retrieved document provide the information this question asks about? "
                + criterion.short_question,
            }
            for criterion in query.rubric
        }
        document = _trimmed(document, self.config.max_document_chars)
        return super()._build_message(query, document).model_copy(update={"questions": questions})

    def _process_answer(self, llm_response: LLMResponseType[BaseModel], query: Query) -> LLMResponseType[BaseModel]:
        addressed = {
            criterion.criterion_name: self._jev_answer(llm_response, criterion.criterion_name).is_yes
            for criterion in query.rubric
        }
        verdicts = self._rubric_schema(query)(reasoning=JEV_REASONING, **addressed)
        return super()._process_answer(
            LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=verdicts), query
        )
