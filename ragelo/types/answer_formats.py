from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Self

PairwiseWinner = Literal["A", "B", "C"]
PairwiseCriterionWinner = Literal["A", "B", "C", "D"]


def swap_pairwise_winner(winner: PairwiseWinner) -> PairwiseWinner:
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return "C"


def swap_criterion_winner(winner: PairwiseCriterionWinner) -> PairwiseCriterionWinner:
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return winner


def swap_pairwise_labels(text: str) -> str:
    swaps = {
        "[[A]]": "[[B]]",
        "[[B]]": "[[A]]",
        "agent A": "agent B",
        "agent B": "agent A",
        "Agent A": "Agent B",
        "Agent B": "Agent A",
        "assistant A": "assistant B",
        "assistant B": "assistant A",
        "Assistant A": "Assistant B",
        "Assistant B": "Assistant A",
        "answer A": "answer B",
        "answer B": "answer A",
        "Answer A": "Answer B",
        "Answer B": "Answer A",
        "response A": "response B",
        "response B": "response A",
        "Response A": "Response B",
        "Response B": "Response A",
    }
    pattern = re.compile(
        r"\[\[A\]\]|\[\[B\]\]|\bagent A\b|\bagent B\b|\bAgent A\b|\bAgent B\b|"
        r"\bassistant A\b|\bassistant B\b|\bAssistant A\b|\bAssistant B\b|\banswer A\b|"
        r"\banswer B\b|\bAnswer A\b|\bAnswer B\b|\bresponse A\b|\bresponse B\b|"
        r"\bResponse A\b|\bResponse B\b"
    )
    return pattern.sub(lambda match: swaps[match.group(0)], text)


class EvaluationAnswer(BaseModel):
    """Base class for LLM-generated evaluation content (without metadata)."""

    pass


class RetrievalEvaluationAnswer(EvaluationAnswer):
    """Output format for evaluating the relevance of a document to a question."""

    reasoning: str = Field(
        ...,
        description="A concise explanation and reasoning of the relevance of the document.",
    )
    score: float | int = Field(
        ...,
        description="Your relevance score for the document. 0 for non-relevant, 1 for somewhat relevant "
        "and 2 for highly relevant.",
    )


class AnswerEvaluationAnswer(EvaluationAnswer):
    """Output format for evaluating the quality of an answer to a question."""

    reasoning: str = Field(
        ...,
        description="A concise explanation and reasoning of the quality of the answer.",
    )
    score: int = Field(
        ...,
        description=(
            "Your score for the quality of the answer. 0 if the answer does not answer the question, "
            "1 if the answer answers the question but is not very helpful and 2 if the answer answers the question "
            "and is very helpful."
        ),
    )


class PairwiseEvaluationAnswer(EvaluationAnswer):
    """Output format for evaluating the quality of answers from two agents to the same question."""

    answer_a_strengths: list[str] = Field(
        default_factory=list,
        description="Short strengths of [[A]]. Use only [[A]] and [[B]] when referring to the assistants.",
    )
    answer_a_weaknesses: list[str] = Field(
        default_factory=list,
        description="Short weaknesses of [[A]]. Use only [[A]] and [[B]] when referring to the assistants.",
    )
    answer_b_strengths: list[str] = Field(
        default_factory=list,
        description="Short strengths of [[B]]. Use only [[A]] and [[B]] when referring to the assistants.",
    )
    answer_b_weaknesses: list[str] = Field(
        default_factory=list,
        description="Short weaknesses of [[B]]. Use only [[A]] and [[B]] when referring to the assistants.",
    )
    answer_a_analysis: str = Field(..., description="A string with your analysis of the answer from agent A")
    answer_b_analysis: str = Field(..., description="A string with your analysis of the answer from agent B")
    comparison_reasoning: str = Field(
        ..., description="A string with your comparison between the two answers and their differences"
    )
    winner_reasoning: str = Field(
        default="",
        description=(
            "A concise explanation of why the winner was chosen, focused on the deciding factor. "
            "Use only [[A]] and [[B]] when referring to the assistants."
        ),
    )
    winner: PairwiseWinner = Field(
        ...,
        description=(
            "The winner of the pairwise comparison. 'A' if the answer from agent A is better, "
            "'B' if the answer from agent B is better, or 'C' for a tie."
        ),
    )

    def swap_perspective(self) -> Self:
        return self.model_copy(
            update={
                "answer_a_strengths": [swap_pairwise_labels(item) for item in self.answer_b_strengths],
                "answer_a_weaknesses": [swap_pairwise_labels(item) for item in self.answer_b_weaknesses],
                "answer_b_strengths": [swap_pairwise_labels(item) for item in self.answer_a_strengths],
                "answer_b_weaknesses": [swap_pairwise_labels(item) for item in self.answer_a_weaknesses],
                "answer_a_analysis": swap_pairwise_labels(self.answer_b_analysis),
                "answer_b_analysis": swap_pairwise_labels(self.answer_a_analysis),
                "comparison_reasoning": swap_pairwise_labels(self.comparison_reasoning),
                "winner_reasoning": swap_pairwise_labels(self.winner_reasoning),
                "winner": swap_pairwise_winner(self.winner),
            }
        )


class RDNAMEvaluationAnswer(RetrievalEvaluationAnswer):
    """LLM-generated evaluation for RDNAM retrieval tasks."""

    reasoning: Annotated[str, SkipJsonSchema] = ""
    score: float = Field(
        ...,
        description="An number between 0 and 2 representing the score of the document.",
    )
    intent_match: float | None = Field(
        ...,
        description="An number between 0 and 2 representing the match of the document to the query intent.",
    )
    trustworthiness: float | None = Field(
        ..., description="An number between 0 and 2 representing the trustworthiness of the document."
    )


class RDNAMNoAspectsAnswer(RetrievalEvaluationAnswer):
    """Output format for evaluating the relevance of a document to a question."""

    reasoning: Annotated[str, SkipJsonSchema] = ""
    score: float = Field(
        ...,
        description="An number between 0 and 2 representing the overall score of the document.",
    )


class RDNAMMultipleAnnotatorsAnswer(RetrievalEvaluationAnswer):
    """Output format for evaluating the relevance of a document to a question by simulating 5 annotators."""

    score: Annotated[float | int, SkipJsonSchema] = 0.0
    reasoning: Annotated[str, SkipJsonSchema] = ""
    annotator_1: RDNAMEvaluationAnswer
    annotator_2: RDNAMEvaluationAnswer
    annotator_3: RDNAMEvaluationAnswer
    annotator_4: RDNAMEvaluationAnswer
    annotator_5: RDNAMEvaluationAnswer


class RDNAMMultipleAnnotatorsNoAspectsAnswer(RetrievalEvaluationAnswer):
    """Output format for evaluating the relevance of a document to a question by simulating 5 annotators."""

    score: Annotated[float | int, SkipJsonSchema] = 0.0
    reasoning: Annotated[str, SkipJsonSchema] = ""
    annotator_1: RDNAMNoAspectsAnswer
    annotator_2: RDNAMNoAspectsAnswer
    annotator_3: RDNAMNoAspectsAnswer
    annotator_4: RDNAMNoAspectsAnswer
    annotator_5: RDNAMNoAspectsAnswer


class Criterion(BaseModel):
    criterion_name: str = Field(
        description="The name of the criterion to be used to evaluate the quality of the responses."
    )
    evidence: list[str] = Field(
        description="The list of documents IDs or snippets that support the criterion. "
        "If no documents support the criterion, leave this list empty."
    )
    short_question: str = Field(
        description="A short, yes/no question that can be used to evaluate the quality of the responses."
    )
    weight: float | None = Field(
        default=None,
        description="The relative importance of this criterion. "
        "Higher values give more weight in the final score. If not provided, all criteria are weighted equally.",
    )


class CriterionEvaluation(BaseModel):
    criterion: Criterion = Field(..., description="The criterion used for evaluating the answer quality")
    agent_a_assessment: str = Field(
        default="",
        description="How well [[A]] satisfies the criterion. Use only [[A]] and [[B]] in text.",
    )
    agent_b_assessment: str = Field(
        default="",
        description="How well [[B]] satisfies the criterion. Use only [[A]] and [[B]] in text.",
    )
    winner_reasoning: str = Field(
        ...,
        description="A brief explanation of why the winner was chosen for this criterion. "
        "Use only [[A]] and [[B]] in text.",
    )
    winner: PairwiseCriterionWinner = Field(..., description="The winner of the criteria")
    score_a: float = Field(default=0.0, description="How well [[A]] satisfies this criterion (0.0 to 1.0).")
    score_b: float = Field(default=0.0, description="How well [[B]] satisfies this criterion (0.0 to 1.0).")
    loser_fix: str = Field(
        default="",
        description="A concise, actionable suggestion for how the losing answer could improve on this criterion.",
    )
    failure_tags: list[str] = Field(
        default_factory=list,
        description="Tags describing the loser's weaknesses (e.g. missing_evidence, unsupported_claim, "
        "incomplete_coverage, poor_synthesis, citation_error, verbosity_without_content).",
    )
    confidence: float = Field(default=1.0, description="Judge's confidence in this criterion's verdict (0.0 to 1.0).")
    missing_evidence_doc_ids: list[str] = Field(default_factory=list, description="Document IDs the loser omitted.")
    missing_evidence_snippets: list[str] = Field(
        default_factory=list, description="Evidence snippets the loser missed."
    )

    def swap_perspective(self) -> Self:
        return self.model_copy(
            update={
                "agent_a_assessment": swap_pairwise_labels(self.agent_b_assessment),
                "agent_b_assessment": swap_pairwise_labels(self.agent_a_assessment),
                "winner_reasoning": swap_pairwise_labels(self.winner_reasoning),
                "winner": swap_criterion_winner(self.winner),
                "score_a": self.score_b,
                "score_b": self.score_a,
                "loser_fix": swap_pairwise_labels(self.loser_fix),
            }
        )

    def merge_with_canonicalized(self, other: Self) -> Self:
        """Merge this criterion evaluation with another in the same A-vs-B perspective.

        Used to reconcile bidirectional pairwise judgments after the reversed
        evaluation has been canonicalized via :py:meth:`swap_perspective`. The
        per-criterion winner agrees with both directions when they match; if
        they disagree, the merged winner is ``C`` (tie). Numeric scores and
        confidence are averaged; tag/evidence lists are unioned. Free-text
        fields are kept from ``self`` (the forward direction).
        """
        if self.winner == other.winner:
            merged_winner: PairwiseCriterionWinner = self.winner
        else:
            merged_winner = "C"
        return self.model_copy(
            update={
                "winner": merged_winner,
                "score_a": (self.score_a + other.score_a) / 2.0,
                "score_b": (self.score_b + other.score_b) / 2.0,
                "confidence": (self.confidence + other.confidence) / 2.0,
                "failure_tags": _union_preserving_order(self.failure_tags, other.failure_tags),
                "missing_evidence_doc_ids": _union_preserving_order(
                    self.missing_evidence_doc_ids, other.missing_evidence_doc_ids
                ),
                "missing_evidence_snippets": _union_preserving_order(
                    self.missing_evidence_snippets, other.missing_evidence_snippets
                ),
            }
        )


def _union_preserving_order(a: list[str], b: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in (*a, *b):
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class EvidenceSnippetEvaluation(BaseModel):
    snippet: str = Field(..., description="The evidence snippet being checked")
    present: bool = Field(..., description="Whether the snippet is present in the answer")
    reasoning: str = Field(..., description="Brief explanation of the presence assessment")


class EvidenceRecallResult(BaseModel):
    snippet_evaluations: list[EvidenceSnippetEvaluation] = Field(..., description="Per-snippet presence evaluations")
    snippets_found: int = Field(..., description="Number of evidence snippets found in the answer")
    total_snippets: int = Field(..., description="Total number of evidence snippets checked")
    recall: float = Field(..., description="Recall ratio (snippets_found / total_snippets)")


class ClaimEvaluation(BaseModel):
    claim: str = Field(..., description="A claim extracted from the answer")
    has_citation: bool = Field(..., description="Whether the claim is supported by a citation")


class CitationExcerptEvaluation(BaseModel):
    citation: str = Field(..., description="A citation found in the answer")
    has_relevant_excerpt: bool = Field(
        ..., description="Whether the citation includes a relevant excerpt from the source"
    )


class CitationQualityResult(BaseModel):
    claim_evaluations: list[ClaimEvaluation] = Field(..., description="Per-claim citation evaluations")
    citation_evaluations: list[CitationExcerptEvaluation] = Field(..., description="Per-citation excerpt evaluations")
    claims_with_citations_ratio: float = Field(..., description="Proportion of claims that have citations")
    citations_with_excerpts_ratio: float = Field(
        ..., description="Proportion of citations that include relevant excerpts"
    )


class EvidenceRecallSchema(BaseModel):
    evaluations: list[EvidenceSnippetEvaluation] = Field(..., description="Per-snippet presence evaluations")


class CitationQualitySchema(BaseModel):
    claims: list[ClaimEvaluation] = Field(..., description="Claims extracted from the answer")
    citations: list[CitationExcerptEvaluation] = Field(..., description="Citations found in the answer")


class RubricAnswerFormat(EvaluationAnswer):
    criteria: list[CriterionEvaluation] = Field(..., description="The criteria used for evaluating the answer quality")
    agent_a_wins: float = Field(..., description="The weighted score of criteria that agent A wins")
    agent_b_wins: float = Field(..., description="The weighted score of criteria that agent B wins")
    equally_good: float = Field(
        ..., description="The weighted score of criteria that agent A and agent B are equally good"
    )
    equally_bad: float = Field(
        ..., description="The weighted score of criteria that agent A and agent B are equally bad"
    )
    winner: PairwiseWinner = Field(..., description="The winner of the pairwise comparison")
    margin: float = Field(default=0.0, description="agent_a_wins minus agent_b_wins (signed)")
    mean_confidence: float = Field(default=1.0, description="Average confidence across all criteria")
    evidence_recall_a: EvidenceRecallResult | None = Field(
        default=None, description="Evidence recall result for agent A"
    )
    evidence_recall_b: EvidenceRecallResult | None = Field(
        default=None, description="Evidence recall result for agent B"
    )
    citation_quality_a: CitationQualityResult | None = Field(
        default=None, description="Citation quality result for agent A"
    )
    citation_quality_b: CitationQualityResult | None = Field(
        default=None, description="Citation quality result for agent B"
    )

    def swap_perspective(self) -> Self:
        return self.model_copy(
            update={
                "criteria": [criterion.swap_perspective() for criterion in self.criteria],
                "agent_a_wins": self.agent_b_wins,
                "agent_b_wins": self.agent_a_wins,
                "winner": swap_pairwise_winner(self.winner),
                "margin": -self.margin,
                "evidence_recall_a": self.evidence_recall_b,
                "evidence_recall_b": self.evidence_recall_a,
                "citation_quality_a": self.citation_quality_b,
                "citation_quality_b": self.citation_quality_a,
            }
        )

    def merge_with_canonicalized(self, other: Self) -> Self:
        """Merge this rubric answer with another in the same A-vs-B perspective.

        Used to reconcile bidirectional pairwise judgments after the reversed
        evaluation has been canonicalized via :py:meth:`swap_perspective`.
        Aggregate scores (``agent_a_wins``, ``agent_b_wins``, ``equally_good``,
        ``equally_bad``, ``mean_confidence``) are averaged across both
        directions; ``margin`` and ``winner`` are recomputed from the merged
        aggregates so they remain consistent with each other and with the
        per-criterion verdicts. Per-criterion records are merged by
        ``criterion_name`` via :py:meth:`CriterionEvaluation.merge_with_canonicalized`;
        criteria that appear in only one direction are kept as-is. Free-text
        fields are kept from ``self`` (the forward direction).
        """
        merged_a_wins = (self.agent_a_wins + other.agent_a_wins) / 2.0
        merged_b_wins = (self.agent_b_wins + other.agent_b_wins) / 2.0
        merged_equally_good = (self.equally_good + other.equally_good) / 2.0
        merged_equally_bad = (self.equally_bad + other.equally_bad) / 2.0
        merged_margin = merged_a_wins - merged_b_wins
        if merged_margin > 0:
            merged_winner: PairwiseWinner = "A"
        elif merged_margin < 0:
            merged_winner = "B"
        else:
            merged_winner = "C"

        other_by_name = {c.criterion.criterion_name: c for c in other.criteria}
        merged_criteria: list[CriterionEvaluation] = []
        for crit in self.criteria:
            counterpart = other_by_name.get(crit.criterion.criterion_name)
            if counterpart is None:
                merged_criteria.append(crit)
            else:
                merged_criteria.append(crit.merge_with_canonicalized(counterpart))

        return self.model_copy(
            update={
                "criteria": merged_criteria,
                "agent_a_wins": merged_a_wins,
                "agent_b_wins": merged_b_wins,
                "equally_good": merged_equally_good,
                "equally_bad": merged_equally_bad,
                "winner": merged_winner,
                "margin": merged_margin,
                "mean_confidence": (self.mean_confidence + other.mean_confidence) / 2.0,
            }
        )


class CriterionEvaluationPointwise(BaseModel):
    criterion: Criterion = Field(..., description="The criterion used for evaluating the answer quality")
    reasoning: str = Field(..., description="The LLM reasoning for the score of the criterion")
    fulfillment: bool | float = Field(..., description="Whether/how much the criterion is fulfilled by the answer")


class RubricPointwiseAnswerFormat(EvaluationAnswer):
    criteria: list[CriterionEvaluationPointwise] = Field(
        ..., description="The criteria used for evaluating the answer quality"
    )
    average_score: float = Field(..., description="The average score of the criteria")
    evidence_recall: EvidenceRecallResult | None = Field(default=None, description="Evidence recall evaluation result")
    citation_quality: CitationQualityResult | None = Field(
        default=None, description="Citation quality evaluation result"
    )


class RubricSchema(BaseModel):
    criteria: list[Criterion] = Field(description="The criteria to be used to evaluate the quality of the responses.")
