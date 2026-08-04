from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field, create_model

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import AnswerEvaluatorFactory, BaseAnswerEvaluator
from ragelo.evaluators.answer_evaluators.builtin_criteria import (
    evaluate_citation_quality,
    evaluate_evidence_recall,
    get_evidence_snippets,
)
from ragelo.evaluators.answer_evaluators.rubric_evaluator_mixin import RubricEvaluatorMixin
from ragelo.types.answer_formats import Criterion, CriterionEvaluationPointwise, RubricPointwiseAnswerFormat
from ragelo.types.configurations import RubricPointwiseEvaluatorConfig
from ragelo.types.evaluables import AgentAnswer, Evaluable
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.query import Query
from ragelo.types.results import AnswerEvaluatorResult, PairwiseGameEvaluatorResult
from ragelo.types.types import AnswerEvaluatorTypes
from ragelo.utils import string_to_template


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.RUBRIC_POINTWISE)
class RubricPointwiseEvaluator(
    RubricEvaluatorMixin, BaseAnswerEvaluator[RubricPointwiseEvaluatorConfig, AnswerEvaluatorResult]
):
    config: RubricPointwiseEvaluatorConfig
    result_type = AnswerEvaluatorResult

    system_prompt = string_to_template(
        """
        You are a domain expert in {{ expert_in }}.{% if company %} You work for {{ company }}.{% endif %} 
        You are tasked with evaluating the quality of a report written by a deep research agent in response of a user's question.
        The report was written based on a set of documents retrieved by the agent, and should thoroughly answer the user's question based exclusively on the relevant documents retrieved by the agent.

        To properly evaluate the quality of the report, you will be provided with a list of criteria to evaluate its quality.
        Each criterion includes a short question and an optional list of documents that support the inclusion of the criterion in the report.
        {% if graduated_scoring %}For each criterion, you should think carefully about how well the report addresses the criterion, provide a brief reasoning, and assign a score from 0 to {{ max_score }}, where 0 means the criterion is not addressed at all and {{ max_score }} means it is fully and thoroughly addressed.{% else %}For each criterion, you should think carefully about whether the report answers the criterion, and include a brief reasoning for your decision.{% endif %}

        You should think carefully about the criteria and the answers, and assign the {% if graduated_scoring %}scores{% else %}final judgement{% endif %} accordingly.

        ## Criteria
        {% for criteria in rubric %}
        Criterion: {{criteria.criterion_name}}
        Supporting Documents: {{criteria.evidence}}
        Short Question: {{criteria.short_question}}
        --------------------------------
        {% endfor %}
        """  # noqa: E501
    )

    user_prompt = string_to_template("""
        [User Question]
            {{query.query}}

        [Agent's Report]
            {{ answer.text }}
        """)

    def _build_evaluation_schema(self, rubric: list[Criterion]) -> Type[BaseModel]:
        criteria_models = {}
        for criterion in rubric:
            if self.config.graduated_scoring:
                criteria_models[criterion.criterion_name] = create_model(
                    criterion.criterion_name,
                    reasoning=(
                        str,
                        Field(
                            description="A brief explanation about your judgement, "
                            "and why you believe the report fulfills or not the criterion"
                        ),
                    ),
                    score=(
                        int,
                        Field(
                            description=f"Score from 0 to {self.config.max_score}. "
                            f"0 means not addressed at all, {self.config.max_score} means fully addressed.",
                            ge=0,
                            le=self.config.max_score,
                        ),
                    ),
                )
            else:
                criteria_models[criterion.criterion_name] = create_model(
                    criterion.criterion_name,
                    reasoning=(
                        str,
                        Field(
                            description="A brief explanation about your judgement, "
                            "and why you believe the report fulfills or not the criterion"
                        ),
                    ),
                    fulfillment=(
                        bool,
                        Field(description="Whether the report fulfills the criterion or not"),
                    ),
                )
        return create_model("EvaluationSchema", **criteria_models)  # type: ignore[call-overload]

    def _build_message(self, query: Query, answer: AgentAnswer) -> LLMInputPrompt:
        system_prompt = self.system_prompt.render(
            expert_in=self.config.expert_in,
            rubric=self._rubric_for(query),
            company=self.config.company,
            graduated_scoring=self.config.graduated_scoring,
            max_score=self.config.max_score,
        )
        user_prompt = self.user_prompt.render(query=query, answer=answer)
        return LLMInputPrompt(
            system_prompt=system_prompt,
            user_message=user_prompt,
            llm_response_schema=self._rubric_schema(query),
        )

    def _process_answer(self, llm_response: LLMResponseType, query: Query) -> LLMResponseType:
        response_dict = llm_response.parsed_answer.model_dump()
        criteria: list[CriterionEvaluationPointwise] = []
        weighted_sum = 0.0
        total_weight = 0.0
        for crit, response in response_dict.items():
            crit_obj = [x for x in self._rubric_for(query) if x.criterion_name == crit][0]
            weight = crit_obj.weight if crit_obj.weight is not None else 1.0
            if self.config.graduated_scoring:
                raw_score = response["score"]
                normalized = raw_score / self.config.max_score
                fulfillment: bool | float = normalized
            else:
                fulfillment = response["fulfillment"]
                normalized = float(fulfillment)
            criterion = CriterionEvaluationPointwise(
                criterion=crit_obj,
                reasoning=response["reasoning"],
                fulfillment=fulfillment,
            )
            criteria.append(criterion)
            weighted_sum += normalized * weight
            total_weight += weight
        return LLMResponseType(
            raw_answer=llm_response.raw_answer,
            parsed_answer=RubricPointwiseAnswerFormat(
                criteria=criteria,
                average_score=weighted_sum / total_weight if total_weight > 0 else 0.0,
                rubric_fingerprint=query.rubric_fingerprint,
            ),
        )

    async def evaluate_async(
        self, eval_sample: tuple[Query, Evaluable]
    ) -> AnswerEvaluatorResult | PairwiseGameEvaluatorResult:
        query, evaluable = eval_sample
        if isinstance(evaluable, AgentAnswer):
            await self._prepare_rubric(query)

        result = await super().evaluate_async(eval_sample)
        if not isinstance(result, AnswerEvaluatorResult) or result.answer is None or result.exception:
            return result

        if not isinstance(evaluable, AgentAnswer) or not evaluable.text:
            return result

        answer_format = result.answer
        if not isinstance(answer_format, RubricPointwiseAnswerFormat):
            return result

        weighted_sum = answer_format.average_score * sum(
            (c.criterion.weight if c.criterion.weight is not None else 1.0) for c in answer_format.criteria
        )
        total_weight = sum(
            (c.criterion.weight if c.criterion.weight is not None else 1.0) for c in answer_format.criteria
        )

        evidence_recall_result = None
        citation_quality_result = None

        if self.config.evidence_recall:
            snippets = get_evidence_snippets(query, self.config.evidence_snippets)
            evidence_recall_result = await evaluate_evidence_recall(self.llm_provider, evaluable.text, snippets)
            weighted_sum += evidence_recall_result.recall * self.config.evidence_recall_weight
            total_weight += self.config.evidence_recall_weight

        if self.config.citation_quality:
            relevant_doc_ids = list(query.retrieved_docs.keys())
            citation_quality_result = await evaluate_citation_quality(
                self.llm_provider, evaluable.text, relevant_doc_ids
            )
            avg_citation_score = (
                citation_quality_result.claims_with_citations_ratio
                + citation_quality_result.citations_with_excerpts_ratio
            ) / 2.0
            weighted_sum += avg_citation_score * self.config.citation_quality_weight
            total_weight += self.config.citation_quality_weight

        updated_answer = answer_format.model_copy(
            update={
                "average_score": weighted_sum / total_weight if total_weight > 0 else 0.0,
                "evidence_recall": evidence_recall_result,
                "citation_quality": citation_quality_result,
            }
        )
        return result.model_copy(update={"answer": updated_answer})
