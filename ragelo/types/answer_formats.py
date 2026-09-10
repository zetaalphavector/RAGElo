from __future__ import annotations

import re
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, computed_field, field_validator
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


@runtime_checkable
class GradedJudgment(Protocol):
    """A judgment that contributes a relevance label to qrels.

    The metrics layer consumes this rather than reaching into a concrete answer format, so a judge
    with a new payload needs no change there. `None` means the judgment carries no usable label and
    the document is left unjudged.
    """

    def relevance(self) -> float | int | None: ...


@runtime_checkable
class RubricJudgment(Protocol):
    """A judgment made against a query's rubric."""

    rubric_fingerprint: str | None


@runtime_checkable
class SubtopicJudgment(Protocol):
    """A judgment that contributes subtopic labels to diversity qrels.

    A query's subtopics are the facets a complete answer must cover, so a document addressing only
    some of them is partial coverage. Coverage measures need one qrel per addressed subtopic.
    """

    def subtopics(self) -> list[str]: ...


class EvaluationAnswer(BaseModel):
    """Base class for LLM-generated evaluation content (without metadata).

    `answer_format` tags the concrete format so that the unions on `EvaluatorResult` subclasses
    can be discriminated instead of inferred from which fields happen to validate. It is hidden
    from the JSON schema, so the LLM is never asked for it, and defaults to empty on formats
    defined outside the library, which fall back to structural matching on load.
    """

    answer_format: SkipJsonSchema[str] = ""


class RetrievalEvaluationAnswer(EvaluationAnswer):
    """Output format for evaluating the relevance of a document to a question."""

    answer_format: SkipJsonSchema[Literal["retrieval_relevance"]] = "retrieval_relevance"

    reasoning: str = Field(
        ...,
        description="A concise explanation and reasoning of the relevance of the document.",
    )
    score: float | int = Field(
        ...,
        description="Your relevance score for the document. 0 for non-relevant, 1 for somewhat relevant "
        "and 2 for highly relevant.",
    )

    def relevance(self) -> float | int | None:
        return self.score


class AnswerEvaluationAnswer(EvaluationAnswer):
    """Output format for evaluating the quality of an answer to a question."""

    answer_format: SkipJsonSchema[Literal["answer_quality"]] = "answer_quality"

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

    answer_format: SkipJsonSchema[Literal["pairwise"]] = "pairwise"

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

    # Narrowing the discriminator in a subclass is intended: an RDNAM payload must not
    # identify itself as the base relevance format.
    answer_format: SkipJsonSchema[Literal["rdnam"]] = "rdnam"  # type: ignore[assignment]

    reasoning: SkipJsonSchema[str] = ""
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

    # Narrowing the discriminator in a subclass is intended: an RDNAM payload must not
    # identify itself as the base relevance format.
    answer_format: SkipJsonSchema[Literal["rdnam_no_aspects"]] = "rdnam_no_aspects"  # type: ignore[assignment]

    reasoning: SkipJsonSchema[str] = ""
    score: float = Field(
        ...,
        description="An number between 0 and 2 representing the overall score of the document.",
    )


class RDNAMMultipleAnnotatorsAnswer(RetrievalEvaluationAnswer):
    """Output format for evaluating the relevance of a document to a question by simulating 5 annotators."""

    # Narrowing the discriminator in a subclass is intended: an RDNAM payload must not
    # identify itself as the base relevance format.
    answer_format: SkipJsonSchema[Literal["rdnam_multiple_annotators"]] = "rdnam_multiple_annotators"  # type: ignore[assignment]

    score: SkipJsonSchema[float | int] = 0.0
    reasoning: SkipJsonSchema[str] = ""
    annotator_1: RDNAMEvaluationAnswer
    annotator_2: RDNAMEvaluationAnswer
    annotator_3: RDNAMEvaluationAnswer
    annotator_4: RDNAMEvaluationAnswer
    annotator_5: RDNAMEvaluationAnswer


class RDNAMMultipleAnnotatorsNoAspectsAnswer(RetrievalEvaluationAnswer):
    """Output format for evaluating the relevance of a document to a question by simulating 5 annotators."""

    # Narrowing the discriminator in a subclass is intended: an RDNAM payload must not
    # identify itself as the base relevance format.
    answer_format: SkipJsonSchema[Literal["rdnam_multiple_annotators_no_aspects"]] = (
        "rdnam_multiple_annotators_no_aspects"  # type: ignore[assignment]
    )

    score: SkipJsonSchema[float | int] = 0.0
    reasoning: SkipJsonSchema[str] = ""
    annotator_1: RDNAMNoAspectsAnswer
    annotator_2: RDNAMNoAspectsAnswer
    annotator_3: RDNAMNoAspectsAnswer
    annotator_4: RDNAMNoAspectsAnswer
    annotator_5: RDNAMNoAspectsAnswer


class Criterion(BaseModel):
    criterion_name: str = Field(
        description="The name of the criterion to be used to evaluate the quality of the responses. "
        "Normalized to a valid Python identifier, as it is used as a response-schema field name "
        "and as a subtopic id in diversity qrels."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="The list of documents IDs or snippets that support the criterion. "
        "If no documents support the criterion, leave this list empty.",
    )
    short_question: str = Field(
        description="A short, yes/no question that can be used to evaluate the quality of the responses."
    )
    weight: float | None = Field(
        default=None,
        description="The relative importance of this criterion. "
        "Higher values give more weight in the final score. If not provided, all criteria are weighted equally.",
    )

    @property
    def effective_weight(self) -> float:
        return 1.0 if self.weight is None else self.weight

    @field_validator("criterion_name", mode="after")
    @classmethod
    def normalize_criterion_name(cls, name: str) -> str:
        normalized = re.sub(r"\W+", "_", name.strip()).strip("_")
        if normalized and normalized[0].isdigit():
            normalized = f"c_{normalized}"
        if not normalized:
            raise ValueError(f"criterion_name {name!r} has no alphanumeric characters to build an identifier from")
        return normalized


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
    answer_format: SkipJsonSchema[Literal["rubric_pairwise"]] = "rubric_pairwise"
    rubric_fingerprint: SkipJsonSchema[str | None] = Field(
        default=None, description="Fingerprint of the rubric this judgment was made against."
    )
    criteria: list[CriterionEvaluation] = Field(..., description="The criteria used for evaluating the answer quality")
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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def agent_a_wins(self) -> float:
        return self.__weight_of("A")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def agent_b_wins(self) -> float:
        return self.__weight_of("B")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def equally_good(self) -> float:
        return self.__weight_of("C")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def equally_bad(self) -> float:
        return self.__weight_of("D")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def margin(self) -> float:
        return self.agent_a_wins - self.agent_b_wins

    @computed_field  # type: ignore[prop-decorator]
    @property
    def winner(self) -> PairwiseWinner:
        if self.margin > 0:
            return "A"
        return "B" if self.margin < 0 else "C"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def mean_confidence(self) -> float:
        if not self.criteria:
            return 1.0
        return sum(c.confidence for c in self.criteria) / len(self.criteria)

    def __weight_of(self, winner: PairwiseCriterionWinner) -> float:
        return sum(c.criterion.effective_weight for c in self.criteria if c.winner == winner)

    def swap_perspective(self) -> Self:
        return self.model_copy(
            update={
                "criteria": [criterion.swap_perspective() for criterion in self.criteria],
                "evidence_recall_a": self.evidence_recall_b,
                "evidence_recall_b": self.evidence_recall_a,
                "citation_quality_a": self.citation_quality_b,
                "citation_quality_b": self.citation_quality_a,
            }
        )

    def merge_with_canonicalized(self, other: Self) -> Self:
        """Merge this rubric answer with another in the same A-vs-B perspective.

        Used to reconcile bidirectional pairwise judgments after the reversed evaluation has been
        canonicalized via :py:meth:`swap_perspective`. Per-criterion records are merged by
        ``criterion_name`` via :py:meth:`CriterionEvaluation.merge_with_canonicalized`; criteria
        that appear in only one direction are kept as-is. Free-text fields are kept from ``self``
        (the forward direction). The aggregates follow from the merged criteria.
        """
        other_by_name = {c.criterion.criterion_name: c for c in other.criteria}
        merged_criteria: list[CriterionEvaluation] = []
        for crit in self.criteria:
            counterpart = other_by_name.get(crit.criterion.criterion_name)
            if counterpart is None:
                merged_criteria.append(crit)
            else:
                merged_criteria.append(crit.merge_with_canonicalized(counterpart))
        return self.model_copy(update={"criteria": merged_criteria})


class CriterionEvaluationPointwise(BaseModel):
    criterion: Criterion = Field(..., description="The criterion used for evaluating the answer quality")
    reasoning: str = Field(..., description="The LLM reasoning for the score of the criterion")
    fulfillment: bool | float = Field(..., description="Whether/how much the criterion is fulfilled by the answer")


class RubricPointwiseAnswerFormat(EvaluationAnswer):
    answer_format: SkipJsonSchema[Literal["rubric_pointwise"]] = "rubric_pointwise"
    rubric_fingerprint: SkipJsonSchema[str | None] = Field(
        default=None, description="Fingerprint of the rubric this judgment was made against."
    )
    criteria: list[CriterionEvaluationPointwise] = Field(
        ..., description="The criteria used for evaluating the answer quality"
    )
    evidence_recall: EvidenceRecallResult | None = Field(default=None, description="Evidence recall evaluation result")
    citation_quality: CitationQualityResult | None = Field(
        default=None, description="Citation quality evaluation result"
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def average_score(self) -> float:
        total_weight = sum(c.criterion.effective_weight for c in self.criteria)
        if total_weight == 0:
            return 0.0
        weighted = sum(float(c.fulfillment) * c.criterion.effective_weight for c in self.criteria)
        return weighted / total_weight


class RubricSchema(BaseModel):
    criteria: list[Criterion] = Field(description="The criteria to be used to evaluate the quality of the responses.")


class RubricCoverageAnswerFormat(EvaluationAnswer):
    """Output format for judging which of a query's rubric criteria a retrieved document addresses.

    The LLM answers one criterion at a time through a per-query response schema, so
    `criteria_addressed` is assembled by the evaluator rather than free-listed by the LLM.
    `score` is the number of criteria addressed, which makes the flat qrels read as a graded
    relevance label while `criteria_addressed` feeds subtopic qrels for coverage measures.
    """

    answer_format: SkipJsonSchema[Literal["rubric_coverage"]] = "rubric_coverage"
    rubric_fingerprint: SkipJsonSchema[str | None] = Field(
        default=None, description="Fingerprint of the rubric this judgment was made against."
    )

    reasoning: str = Field(description="A concise explanation of which criteria the document addresses.")
    criteria_addressed: list[str] = Field(
        default_factory=list, description="The names of the rubric criteria this document addresses."
    )
    criteria: list[CriterionEvaluationPointwise] = Field(
        default_factory=list,
        description="The per-criterion judgement. `fulfillment` is the addressed flag, or the score "
        "divided by `max_score` under graduated scoring.",
    )
    score: SkipJsonSchema[float | int] = 0

    def relevance(self) -> float | int | None:
        return self.score

    def subtopics(self) -> list[str]:
        return self.criteria_addressed
