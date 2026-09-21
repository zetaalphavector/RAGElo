import logging

from pydantic import BaseModel

from ragelo.evaluators.jev_evaluator_mixin import JevEvaluatorMixin, JevRubricEvaluatorMixin
from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import RetrievalEvaluatorFactory
from ragelo.evaluators.retrieval_evaluators.rdnam_evaluator import RDNAMEvaluator
from ragelo.evaluators.retrieval_evaluators.reasoner_evaluator import ReasonerEvaluator
from ragelo.evaluators.retrieval_evaluators.rubric_coverage_evaluator import RubricCoverageEvaluator
from ragelo.types import Document, LLMInputPrompt, LLMResponseType, Query
from ragelo.types.answer_formats import RDNAMEvaluationAnswer, RetrievalEvaluationAnswer
from ragelo.types.configurations import (
    JevRDNAMEvaluatorConfig,
    JevRetrievalEvaluatorConfig,
    JevRubricCoverageEvaluatorConfig,
)
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
class JevRetrievalEvaluator(JevEvaluatorMixin, ReasonerEvaluator):
    """Asks Jev one yes/no question about the document, or with `boolean_question=False` a score
    question over the relevance grades. The yes-probability is scaled to the top grade, so on a 0-2
    scale a document rounds to grade 2 from a probability of 0.75 and to grade 1 from 0.25.
    """

    config: JevRetrievalEvaluatorConfig
    system_prompt = string_to_template("How relevant is the document to the user question?")

    boolean_prompt = (
        "Assume you are writing a report on the topic of the user question. "
        "Would you use any of the information contained in the document in that report?"
    )

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        prompt = super()._build_message(query, _trimmed(document, self.config.max_document_chars))
        if self.config.boolean_question:
            instructions = prompt.system_prompt if self.config.system_prompt else self.boolean_prompt
            return prompt.model_copy(
                update={
                    "system_prompt": instructions,
                    "questions": {"score": {"type": "boolean"}},
                    "batch_key": self._batch_key(query),
                }
            )
        question = {"type": "score", "criteria": list(self.relevance_grades)}
        return prompt.model_copy(update={"questions": {"score": question}, "batch_key": self._batch_key(query)})

    def _process_answer(self, llm_response: LLMResponseType[BaseModel], query: Query) -> LLMResponseType[BaseModel]:
        answer = self._jev_answer(llm_response, "score")
        parsed = RetrievalEvaluationAnswer(
            reasoning=self._reasoning(answer),
            score=self._score(answer),
            probabilities=answer.probabilities,
            confidence=answer.confidence,
        )
        return LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=parsed)

    def _score(self, answer: JevAnswer) -> float:
        if answer.probability is not None:
            return answer.probability * self.max_score
        return int(answer.label)

    def _reasoning(self, answer: JevAnswer) -> str:
        """Answer evaluators quote a document's relevance reasoning in their prompts, so it says what Jev found."""
        if answer.probability is not None:
            return f"Jev gives a probability of {answer.probability:.2f} that the document is relevant."
        return f"Jev's most likely relevance grade: {self.relevance_grades[int(answer.label)]}"


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.JEV_RDNAM)
class JevRDNAMEvaluator(JevEvaluatorMixin, RDNAMEvaluator):
    """RDNAM's prompt and relevance grades as Jev score questions, one for the relevance and one per aspect.

    The stored scores are Jev's expected levels, so they are fractional like RDNAM's annotator averages.
    Jev answers every question on its own, so the aspects are reported without informing the relevance.
    """

    config: JevRDNAMEvaluatorConfig
    unsupported_options = ("use_multiple_annotators",)

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        prompt = super()._build_message(query, _trimmed(document, self.config.max_document_chars))
        levels = [f"{level} out of {self.max_score}" for level in range(self.max_score + 1)]
        questions: dict[str, dict[str, object]] = {
            name: {"type": "score", "instructions": f"Rate {aspect}.", "criteria": levels}
            for name, aspect in self.aspects.items()
        }
        questions["score"] = {"type": "score", "criteria": list(self.relevance_grades)}
        return prompt.model_copy(update={"questions": questions, "batch_key": self._batch_key(query)})

    def _process_answer(self, llm_response: LLMResponseType[BaseModel], query: Query) -> LLMResponseType[BaseModel]:
        relevance = self._jev_answer(llm_response, "score")
        answer = RDNAMEvaluationAnswer(
            reasoning=f"Jev's expected relevance grade is {relevance.score:.2f} out of {self.max_score}.",
            probabilities=relevance.probabilities,
            confidence=relevance.confidence,
            **{name: self._jev_answer(llm_response, name).score for name in [*self.aspects, "score"]},
        )
        return LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=answer)


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
        prompt = super()._build_message(query, document)
        return prompt.model_copy(update={"questions": questions, "batch_key": self._batch_key(query)})

    def _process_answer(self, llm_response: LLMResponseType[BaseModel], query: Query) -> LLMResponseType[BaseModel]:
        addressed = {
            criterion.criterion_name: self._jev_answer(llm_response, criterion.criterion_name).is_yes
            for criterion in query.rubric
        }
        found = ", ".join(name for name, is_addressed in addressed.items() if is_addressed) or "none"
        verdicts = self._rubric_schema(query)(reasoning=f"Criteria Jev found addressed: {found}", **addressed)
        processed = super()._process_answer(
            LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=verdicts), query
        )
        for judged in processed.parsed_answer.criteria:
            judged.probability = self._jev_answer(llm_response, judged.criterion.criterion_name).probability
        return processed
