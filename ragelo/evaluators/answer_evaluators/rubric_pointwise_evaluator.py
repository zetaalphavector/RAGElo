from __future__ import annotations

from typing import cast

from pydantic import BaseModel, Field, create_model

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
    BaseAnswerEvaluator,
    T_AnswerResult,
)
from ragelo.evaluators.answer_evaluators.builtin_criteria import (
    citation_quality_criterion,
    citation_quality_score,
    evaluate_citation_quality,
    evaluate_evidence_recall,
    evidence_recall_criterion,
    get_evidence_snippets,
)
from ragelo.evaluators.answer_evaluators.rubric_evaluator_mixin import RubricEvaluatorMixin
from ragelo.types.answer_formats import Criterion, CriterionEvaluationPointwise, RubricPointwiseAnswerFormat
from ragelo.types.configurations import RubricPointwiseEvaluatorConfig
from ragelo.types.evaluables import AgentAnswer, Evaluable
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.query import Query
from ragelo.types.results import AnswerEvaluatorResult
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
        """
    )

    user_prompt = string_to_template("""
        [User Question]
            {{query.query}}

        [Agent's Report]
            {{ answer.text }}
        """)

    def _build_evaluation_schema(self, rubric: list[Criterion]) -> type[BaseModel]:
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
        for crit, response in response_dict.items():
            crit_obj = [x for x in self._rubric_for(query) if x.criterion_name == crit][0]
            if self.config.graduated_scoring:
                fulfillment: bool | float = response["score"] / self.config.max_score
            else:
                fulfillment = response["fulfillment"]
            criteria.append(
                CriterionEvaluationPointwise(
                    criterion=crit_obj,
                    reasoning=response["reasoning"],
                    fulfillment=fulfillment,
                )
            )
        return LLMResponseType(
            raw_answer=llm_response.raw_answer,
            parsed_answer=RubricPointwiseAnswerFormat(
                criteria=criteria,
                rubric_fingerprint=query.rubric_fingerprint,
            ),
        )

    async def _augment_judgment(self, result: T_AnswerResult, query: Query, evaluable: Evaluable) -> T_AnswerResult:
        if not self.config.evidence_recall and not self.config.citation_quality:
            return result
        answer_format = result.answer
        if not isinstance(answer_format, RubricPointwiseAnswerFormat):
            return result
        if not isinstance(evaluable, AgentAnswer) or not evaluable.text:
            return result

        criteria = list(answer_format.criteria)
        evidence_recall_result = None
        citation_quality_result = None

        if self.config.evidence_recall:
            snippets = get_evidence_snippets(query, self.config.evidence_snippets)
            evidence_recall_result = await evaluate_evidence_recall(self.llm_provider, evaluable.text, snippets)
            criteria.append(
                CriterionEvaluationPointwise(
                    criterion=evidence_recall_criterion(self.config.evidence_recall_weight),
                    reasoning=f"{evidence_recall_result.snippets_found} of "
                    f"{evidence_recall_result.total_snippets} evidence snippets are present in the answer.",
                    fulfillment=evidence_recall_result.recall,
                )
            )

        if self.config.citation_quality:
            citation_quality_result = await evaluate_citation_quality(
                self.llm_provider, evaluable.text, list(query.retrieved_docs.keys())
            )
            criteria.append(
                CriterionEvaluationPointwise(
                    criterion=citation_quality_criterion(self.config.citation_quality_weight),
                    reasoning=f"{citation_quality_result.claims_with_citations_ratio:.2f} of claims carry a "
                    f"citation and {citation_quality_result.citations_with_excerpts_ratio:.2f} of citations "
                    "carry a relevant excerpt.",
                    fulfillment=citation_quality_score(citation_quality_result),
                )
            )

        updated_answer = answer_format.model_copy(
            update={
                "criteria": criteria,
                "evidence_recall": evidence_recall_result,
                "citation_quality": citation_quality_result,
            }
        )
        return cast(T_AnswerResult, result.model_copy(update={"answer": updated_answer}))
