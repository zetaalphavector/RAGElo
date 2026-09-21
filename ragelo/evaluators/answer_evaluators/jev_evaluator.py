from typing import ClassVar

from pydantic import BaseModel

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import AnswerEvaluatorFactory, BaseAnswerEvaluator
from ragelo.evaluators.answer_evaluators.pairwise_evaluator import PairwiseAnswerEvaluator
from ragelo.evaluators.answer_evaluators.rubric_pairwise_evaluator import RubricPairwiseEvaluator
from ragelo.evaluators.answer_evaluators.rubric_pointwise_evaluator import RubricPointwiseEvaluator
from ragelo.evaluators.jev_evaluator_mixin import JEV_REASONING, JevEvaluatorMixin, JevRubricEvaluatorMixin
from ragelo.types import AgentAnswer, LLMInputPrompt, LLMResponseType, PairwiseGame, Query
from ragelo.types.answer_formats import AnswerEvaluationAnswer, PairwiseEvaluationAnswer
from ragelo.types.configurations import (
    JevAnswerEvaluatorConfig,
    JevPairwiseEvaluatorConfig,
    JevRubricPairwiseEvaluatorConfig,
    JevRubricPointwiseEvaluatorConfig,
)
from ragelo.types.results import AnswerEvaluatorResult
from ragelo.types.types import AnswerEvaluatorTypes
from ragelo.utils import string_to_template


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.JEV)
class JevAnswerEvaluator(JevEvaluatorMixin, BaseAnswerEvaluator[JevAnswerEvaluatorConfig, AnswerEvaluatorResult]):
    """Scores an answer with one Jev score question whose levels are the answer grades."""

    config: JevAnswerEvaluatorConfig
    result_type = AnswerEvaluatorResult
    answer_format = AnswerEvaluationAnswer
    system_prompt = string_to_template("How well does the answer respond to the user question?")
    user_prompt = string_to_template("""
        [user question]
        {{ query.query }}
        {%- if documents %}

        [retrieved documents]
        {%- for d in documents %}
        [{{ d.did }}] {{ d.text }}
        {%- endfor %}
        {%- endif %}

        [answer]
        {{ answer.text }}
        """)

    def _filter_documents(self, query: Query):
        return super()._filter_documents(query) if self.config.include_raw_documents else []

    def _build_message(self, query: Query, answer: AgentAnswer) -> LLMInputPrompt:
        prompt = super()._build_message(query, answer)
        question = {"type": "score", "criteria": self.config.answer_grades}
        return prompt.model_copy(update={"questions": {"score": question}, "batch_key": self._batch_key(query)})

    def _process_answer(self, llm_response: LLMResponseType[BaseModel], query: Query) -> LLMResponseType[BaseModel]:
        answer = self._jev_answer(llm_response, "score")
        parsed = AnswerEvaluationAnswer(
            reasoning=JEV_REASONING,
            score=int(answer.label),
            probabilities=answer.probabilities,
            confidence=answer.confidence,
        )
        return LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=parsed)


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.JEV_PAIRWISE)
class JevPairwiseEvaluator(JevEvaluatorMixin, PairwiseAnswerEvaluator):
    """Picks the better of two answers with one Jev choice question, over the state the pairwise evaluator builds."""

    config: JevPairwiseEvaluatorConfig
    winner_options: ClassVar[dict[str, str]] = {
        "A": "The answer from assistant A is better.",
        "B": "The answer from assistant B is better.",
        "C": "Both answers are equally good, or equally bad.",
    }
    system_prompt = string_to_template(
        "Which assistant gave the better answer to the user question, considering {{ factors }}?"
    )

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> LLMInputPrompt:
        question = {"type": "choice", "criteria": self.winner_options}
        prompt = super()._build_message_pairwise(query, game)
        return prompt.model_copy(update={"questions": {"winner": question}, "batch_key": self._batch_key(query)})

    def _process_answer(self, llm_response: LLMResponseType[BaseModel], query: Query) -> LLMResponseType[BaseModel]:
        answer = self._jev_answer(llm_response, "winner")
        parsed = PairwiseEvaluationAnswer(
            answer_a_analysis=JEV_REASONING,
            answer_b_analysis=JEV_REASONING,
            comparison_reasoning=JEV_REASONING,
            winner=answer.label,  # type: ignore[arg-type]
            probabilities=answer.probabilities,
            confidence=answer.confidence,
        )
        return LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=parsed)


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.JEV_RUBRIC_POINTWISE)
class JevRubricPointwiseEvaluator(JevRubricEvaluatorMixin, RubricPointwiseEvaluator):
    config: JevRubricPointwiseEvaluatorConfig

    def _build_message(self, query: Query, answer: AgentAnswer) -> LLMInputPrompt:
        questions = {
            criterion.criterion_name: {"type": "boolean", "instructions": criterion.short_question}
            for criterion in self._rubric_for(query)
        }
        prompt = super()._build_message(query, answer)
        return prompt.model_copy(update={"questions": questions, "batch_key": self._batch_key(query)})

    def _process_answer(self, llm_response: LLMResponseType[BaseModel], query: Query) -> LLMResponseType[BaseModel]:
        verdicts = self._rubric_schema(query)(
            **{
                criterion.criterion_name: {
                    "reasoning": JEV_REASONING,
                    "fulfillment": self._jev_answer(llm_response, criterion.criterion_name).is_yes,
                }
                for criterion in self._rubric_for(query)
            }
        )
        processed = super()._process_answer(
            LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=verdicts), query
        )
        for judged in processed.parsed_answer.criteria:
            judged.probability = self._jev_answer(llm_response, judged.criterion.criterion_name).probability
        return processed


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.JEV_RUBRIC_PAIRWISE)
class JevRubricPairwiseEvaluator(JevRubricEvaluatorMixin, RubricPairwiseEvaluator):
    config: JevRubricPairwiseEvaluatorConfig
    winner_options: ClassVar[dict[str, str]] = {
        "A": "Agent A satisfies the criterion better.",
        "B": "Agent B satisfies the criterion better.",
        "C": "Both agents satisfy the criterion equally well.",
        "D": "Neither agent satisfies the criterion.",
    }

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> LLMInputPrompt:
        options = {k: v for k, v in self.winner_options.items() if k != "D" or self.config.preserve_d}
        questions = {
            criterion.criterion_name: {
                "type": "choice",
                "instructions": f"Which agent's answer better satisfies this criterion? {criterion.short_question}",
                "criteria": options,
            }
            for criterion in self._rubric_for(query)
        }
        prompt = super()._build_message_pairwise(query, game)
        return prompt.model_copy(update={"questions": questions, "batch_key": self._batch_key(query)})

    def _process_answer(self, llm_response: LLMResponseType[BaseModel], query: Query) -> LLMResponseType[BaseModel]:
        verdicts = {}
        for criterion in self._rubric_for(query):
            answer = self._jev_answer(llm_response, criterion.criterion_name)
            verdicts[criterion.criterion_name] = {
                "agent_a_assessment": JEV_REASONING,
                "agent_b_assessment": JEV_REASONING,
                "winner_reasoning": JEV_REASONING,
                "winner": answer.label,
            }
        return super()._process_answer(
            LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=self._rubric_schema(query)(**verdicts)),
            query,
        )
