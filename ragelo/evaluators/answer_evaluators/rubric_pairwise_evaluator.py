from __future__ import annotations

from typing import Literal, Type, cast

from pydantic import BaseModel, Field, create_model

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import AnswerEvaluatorFactory, T_AnswerResult
from ragelo.evaluators.answer_evaluators.builtin_criteria import (
    citation_quality_criterion,
    citation_quality_score,
    evaluate_citation_quality,
    evaluate_evidence_recall,
    evidence_recall_criterion,
    get_evidence_snippets,
)
from ragelo.evaluators.answer_evaluators.pairwise_evaluator import PairwiseAnswerEvaluator
from ragelo.evaluators.answer_evaluators.rubric_evaluator_mixin import RubricEvaluatorMixin
from ragelo.types.answer_formats import Criterion, CriterionEvaluation, PairwiseCriterionWinner, RubricAnswerFormat
from ragelo.types.configurations import RubricPairwiseEvaluatorConfig
from ragelo.types.evaluables import ChatMessage, Evaluable, PairwiseGame
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.query import Query
from ragelo.types.types import AnswerEvaluatorTypes
from ragelo.utils import string_to_template


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.RUBRIC_PAIRWISE)
class RubricPairwiseEvaluator(RubricEvaluatorMixin, PairwiseAnswerEvaluator):
    config: RubricPairwiseEvaluatorConfig
    system_prompt = string_to_template(
        """
        You are a domain expert in {{ expert_in }}.{% if company %} You work for {{ company }}.{% endif %} 
        You are tasked with evaluating the quality of two {% if is_conversation %}conversations{% else %}reports{% endif %} written by two agents in response of a user's question.
        The {% if is_conversation %}conversations{% else %}reports{% endif %} are written based on a set of documents retrieved by the two agents, and should answer the user's question based on the relevant documents retrieved by the agents.

        To properly evaluate the quality of the {% if is_conversation %}conversations{% else %}reports{% endif %}, you will be provided with a list of criteria to evaluate the quality of the responses. 
        Each criterion includes a short question and an optional list of documents that support the inclusion of the criterion in the {% if is_conversation %}conversation{% else %}report{% endif %}.
        For each criterion, you should think carefully about which of two answers better answers the criterion, and provide the following:

        ### Winner
        Assign one of the following values:
        - A if the {% if is_conversation %}conversation{% else %}report{% endif %} written by Agent A clearly answers the criterion better than the {% if is_conversation %}conversation{% else %}report{% endif %} written by Agent B.
        - B if the {% if is_conversation %}conversation{% else %}report{% endif %} written by Agent B clearly answers the criterion better than the {% if is_conversation %}conversation{% else %}report{% endif %} written by Agent A.
        - C if the {% if is_conversation %}conversation{% else %}report{% endif %} written by Agent A and Agent B are equally good and answer the criterion equally well.
        {% if preserve_d %}- D if the {% if is_conversation %}conversation{% else %}report{% endif %} written by Agent A and Agent B are equally bad and neither answers the criterion.
        {% endif %}
        {% if rich_output %}
        ### Scores
        - `score_a`: Rate how well [[A]] satisfies this criterion from 0.0 (not at all) to 1.0 (fully).
        - `score_b`: Rate how well [[B]] satisfies this criterion from 0.0 (not at all) to 1.0 (fully).

        ### Diagnostics
        - `loser_fix`: A concise, actionable suggestion for how the losing answer could improve on this criterion. If tied{% if preserve_d %} (C or D){% endif %}, leave empty.
        - `failure_tags`: Tag the loser's weaknesses using zero or more of: `missing_evidence`, `unsupported_claim`, `incomplete_coverage`, `poor_synthesis`, `citation_error`, `verbosity_without_content`. If tied, leave empty.
        - `confidence`: Your confidence in this criterion's verdict from 0.0 (very uncertain) to 1.0 (certain).
        {% endif %}
        {% if include_evidence %}

        ### Evidence Assessment
        - `missing_evidence_doc_ids`: List any document IDs from the supporting documents that the loser failed to use.
        - `missing_evidence_snippets`: List specific evidence snippets from the supporting documents that the loser omitted or misused.
        {% endif %}

        ## Output Constraints
        - In any free-text field, refer to the reports only as [[A]] and [[B]].
        - Do not use phrases such as first answer, second answer, former, latter, this answer, or that answer.
        - For each criterion, provide a side-specific assessment for [[A]], a side-specific assessment for [[B]], and a brief winner_reasoning.

        {% if evidence_snippets %}
        ## Evidence Snippets
        The following evidence snippets were extracted from the retrieved documents. Use them to assess whether each answer correctly leverages the available evidence.
        {% for snippet in evidence_snippets %}
        - {{ snippet }}
        {% endfor %}
        {% endif %}

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

        [The Start of Agent A's Answer]
        {% if game.agent_a_answer.conversation %}
        {% for msg in game.agent_a_answer.conversation %}
        {{ msg }}
        {% endfor %}
        {% else %}
            {{ game.agent_a_answer.text }}
        {% endif %}
        [The End of Agent A's Answer]

        [The Start of Agent B's Answer]
        {% if game.agent_b_answer.conversation %}
        {% for msg in game.agent_b_answer.conversation %}
        {{ msg }}
        {% endfor %}
        {% else %}
            {{ game.agent_b_answer.text }}
        {% endif %}
        [The End of Agent B's Answer]""")

    def _build_evaluation_schema(self, rubric: list[Criterion]) -> Type[BaseModel]:
        include_evidence = self.config.include_evidence_in_evaluation
        rich_output = self.config.rich_pairwise_output
        preserve_d = self.config.preserve_d
        winner_type = Literal["A", "B", "C", "D"] if preserve_d else Literal["A", "B", "C"]
        criteria_models = {}
        for criterion in rubric:
            fields: dict = {
                "agent_a_assessment": (str, Field(description="How well [[A]] satisfies the criterion.")),
                "agent_b_assessment": (str, Field(description="How well [[B]] satisfies the criterion.")),
                "winner_reasoning": (str, Field(description="A brief explanation of why the winner was chosen.")),
                "winner": (winner_type, Field(description="The winner of the criterion")),
            }
            if rich_output:
                fields.update(
                    {
                        "score_a": (float, Field(description="Score for [[A]] on this criterion (0.0 to 1.0).")),
                        "score_b": (float, Field(description="Score for [[B]] on this criterion (0.0 to 1.0).")),
                        "loser_fix": (
                            str,
                            Field(
                                description="One sentence: what should the losing side do differently? Empty if tied."
                            ),
                        ),
                        "failure_tags": (
                            list[str],
                            Field(
                                description="Tags from: missing_evidence, unsupported_claim, incomplete_coverage, "
                                "poor_synthesis, citation_error, verbosity_without_content. Empty if tied."
                            ),
                        ),
                        "confidence": (float, Field(description="Your confidence in this verdict, 0.0 to 1.0.")),
                    }
                )
            if include_evidence:
                fields["missing_evidence_doc_ids"] = (
                    list[str],
                    Field(description="Document IDs the loser failed to use."),
                )
                fields["missing_evidence_snippets"] = (
                    list[str],
                    Field(description="Evidence snippets the loser omitted or misused."),
                )
            criteria_models[criterion.criterion_name] = create_model(criterion.criterion_name, **fields)
        return create_model("EvaluationSchema", **criteria_models)  # type: ignore[call-overload]

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> LLMInputPrompt:
        rubric = self._rubric_for(query)
        include_evidence = self.config.include_evidence_in_evaluation
        evidence_snippets: list[str] = []
        is_conversation = bool(game.agent_a_answer.conversation or game.agent_b_answer.conversation)
        if include_evidence:
            evidence_snippets = get_evidence_snippets(query, self.config.evidence_snippets)
            evidence_snippets = _truncate_snippets(evidence_snippets, self.config.max_evidence_tokens)
        system_prompt = self.system_prompt.render(
            expert_in=self.config.expert_in,
            rubric=rubric,
            company=self.config.company,
            include_evidence=include_evidence,
            is_conversation=is_conversation,
            evidence_snippets=evidence_snippets,
            preserve_d=self.config.preserve_d,
            rich_output=self.config.rich_pairwise_output,
        )
        user_prompt = self.user_prompt.render(
            query=query,
            game=game,
        )
        return LLMInputPrompt(
            system_prompt=system_prompt,
            user_message=user_prompt,
            llm_response_schema=self._rubric_schema(query),
        )

    def _process_answer(self, llm_response: LLMResponseType, query: Query) -> LLMResponseType:
        response_dict = llm_response.parsed_answer.model_dump()
        criteria: list[CriterionEvaluation] = []
        for crit, response in response_dict.items():
            crit_obj = [x for x in self._rubric_for(query) if x.criterion_name == crit][0]
            if len(response["winner"]) > 1:
                response["winner"] = response["winner"][-1]
            if not self.config.preserve_d and response["winner"] == "D":
                response["winner"] = "C"
            criteria.append(
                CriterionEvaluation(
                    criterion=crit_obj,
                    agent_a_assessment=response.get("agent_a_assessment", ""),
                    agent_b_assessment=response.get("agent_b_assessment", ""),
                    winner_reasoning=response.get("winner_reasoning", response.get("reasoning", "")),
                    winner=response["winner"],
                    score_a=response.get("score_a", 0.0),
                    score_b=response.get("score_b", 0.0),
                    loser_fix=response.get("loser_fix", ""),
                    failure_tags=response.get("failure_tags", []),
                    confidence=response.get("confidence", 1.0),
                    missing_evidence_doc_ids=response.get("missing_evidence_doc_ids", []),
                    missing_evidence_snippets=response.get("missing_evidence_snippets", []),
                )
            )
        return LLMResponseType(
            raw_answer=llm_response.raw_answer,
            parsed_answer=RubricAnswerFormat(criteria=criteria, rubric_fingerprint=query.rubric_fingerprint),
        )

    def _rubric_conversation_context(self, query: Query) -> list[ChatMessage]:
        return self._get_conversation_context(query)

    async def _augment_judgment(self, result: T_AnswerResult, query: Query, evaluable: Evaluable) -> T_AnswerResult:
        if not self.config.evidence_recall and not self.config.citation_quality:
            return result
        answer_format = result.answer
        if not isinstance(answer_format, RubricAnswerFormat) or not isinstance(evaluable, PairwiseGame):
            return result

        agent_a_text = evaluable.agent_a_answer.final_response
        agent_b_text = evaluable.agent_b_answer.final_response
        criteria = list(answer_format.criteria)
        updates: dict = {}

        if self.config.evidence_recall:
            snippets = get_evidence_snippets(query, self.config.evidence_snippets)
            recall_a = await evaluate_evidence_recall(self.llm_provider, agent_a_text, snippets)
            recall_b = await evaluate_evidence_recall(self.llm_provider, agent_b_text, snippets)
            updates["evidence_recall_a"] = recall_a
            updates["evidence_recall_b"] = recall_b
            criteria.append(
                self.__side_by_side_criterion(
                    evidence_recall_criterion(self.config.evidence_recall_weight),
                    recall_a.recall,
                    recall_b.recall,
                    f"[[A]] includes {recall_a.snippets_found} of {recall_a.total_snippets} evidence snippets, "
                    f"[[B]] includes {recall_b.snippets_found} of {recall_b.total_snippets}.",
                )
            )

        if self.config.citation_quality:
            relevant_doc_ids = list(query.retrieved_docs.keys())
            cq_a = await evaluate_citation_quality(self.llm_provider, agent_a_text, relevant_doc_ids)
            cq_b = await evaluate_citation_quality(self.llm_provider, agent_b_text, relevant_doc_ids)
            updates["citation_quality_a"] = cq_a
            updates["citation_quality_b"] = cq_b
            score_a = citation_quality_score(cq_a)
            score_b = citation_quality_score(cq_b)
            criteria.append(
                self.__side_by_side_criterion(
                    citation_quality_criterion(self.config.citation_quality_weight),
                    score_a,
                    score_b,
                    f"[[A]] scores {score_a:.2f} on citation quality, [[B]] scores {score_b:.2f}.",
                )
            )

        updates["criteria"] = criteria
        return cast(T_AnswerResult, result.model_copy(update={"answer": answer_format.model_copy(update=updates)}))

    @staticmethod
    def __side_by_side_criterion(
        criterion: Criterion, score_a: float, score_b: float, reasoning: str
    ) -> CriterionEvaluation:
        winner: PairwiseCriterionWinner = "A" if score_a > score_b else "B" if score_b > score_a else "C"
        return CriterionEvaluation(
            criterion=criterion,
            winner=winner,
            winner_reasoning=reasoning,
            score_a=score_a,
            score_b=score_b,
        )


def _truncate_snippets(snippets: list[str], max_chars: int) -> list[str]:
    result = []
    total = 0
    for s in snippets:
        if total + len(s) > max_chars:
            remaining = max_chars - total
            if remaining > 50:
                result.append(s[:remaining])
            break
        result.append(s)
        total += len(s)
    return result
