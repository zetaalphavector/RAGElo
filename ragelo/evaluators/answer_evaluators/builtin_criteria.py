from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ragelo.types.answer_formats import (
    CitationExcerptEvaluation,
    CitationQualityResult,
    CitationQualitySchema,
    ClaimEvaluation,
    EvidenceRecallResult,
    EvidenceRecallSchema,
    EvidenceSnippetEvaluation,
)
from ragelo.types.formats import LLMInputPrompt
from ragelo.utils import string_to_template

if TYPE_CHECKING:
    from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
    from ragelo.types.query import Query

logger = logging.getLogger(__name__)

EVIDENCE_RECALL_SYSTEM_PROMPT = string_to_template(
    """
    You are tasked with evaluating whether specific evidence snippets are present in a report.
    You will be given a list of evidence snippets and a report. For each snippet, determine whether
    the snippet's content is partly or completely present in the report.
    Be thorough: a snippet counts as present if the report conveys the same information,
    even if the exact wording differs.
    """
)

EVIDENCE_RECALL_USER_PROMPT = string_to_template(
    """
    [Evidence Snippets to Check]
    {% for snippet in snippets %}
    {{ loop.index }}. {{ snippet }}
    --------------------------------
    {% endfor %}

    [Report]
    {{ answer_text }}
    """
)

CITATION_QUALITY_SYSTEM_PROMPT = string_to_template(
    """
    You are tasked with evaluating the citation quality of a report.
    You should:
    1. Identify all distinct claims made in the report.
    2. For each claim, determine whether it is supported by a citation (a reference to a source document).
    3. Identify all citations in the report.
    4. For each citation, determine whether it includes a relevant excerpt or quote from the source.

    The following document IDs are considered relevant sources:
    {% for doc_id in relevant_doc_ids %}
    - {{ doc_id }}
    {% endfor %}
    """
)

CITATION_QUALITY_USER_PROMPT = string_to_template(
    """
    [Report]
    {{ answer_text }}
    """
)


def get_evidence_snippets(
    query: Query,
    config_snippets: dict[str, list[str]] | None,
    criteria_cache: dict | None = None,
) -> list[str]:
    if config_snippets and query.qid in config_snippets:
        return config_snippets[query.qid]

    if criteria_cache and query.qid in criteria_cache:
        rubric = criteria_cache[query.qid]
        all_evidence: list[str] = []
        for criterion in rubric.criteria:
            all_evidence.extend(criterion.evidence)
        if all_evidence:
            return all_evidence

    snippets = []
    for doc in query.retrieved_docs.values():
        if doc.text:
            snippets.append(doc.text)
    return snippets


async def evaluate_evidence_recall(
    llm_provider: BaseLLMProvider,
    answer_text: str,
    snippets: list[str],
) -> EvidenceRecallResult:
    if not snippets:
        return EvidenceRecallResult(
            snippet_evaluations=[],
            snippets_found=0,
            total_snippets=0,
            recall=0.0,
        )

    system_prompt = EVIDENCE_RECALL_SYSTEM_PROMPT.render()
    user_prompt = EVIDENCE_RECALL_USER_PROMPT.render(snippets=snippets, answer_text=answer_text)
    llm_input = LLMInputPrompt(system_prompt=system_prompt, user_message=user_prompt)
    llm_response = await llm_provider.call_async(llm_input, response_schema=EvidenceRecallSchema)

    evaluations: list[EvidenceSnippetEvaluation] = llm_response.parsed_answer.evaluations
    snippets_found = sum(1 for e in evaluations if e.present)
    total = len(evaluations)
    return EvidenceRecallResult(
        snippet_evaluations=evaluations,
        snippets_found=snippets_found,
        total_snippets=total,
        recall=snippets_found / total if total > 0 else 0.0,
    )


async def evaluate_citation_quality(
    llm_provider: BaseLLMProvider,
    answer_text: str,
    relevant_doc_ids: list[str],
) -> CitationQualityResult:
    system_prompt = CITATION_QUALITY_SYSTEM_PROMPT.render(relevant_doc_ids=relevant_doc_ids)
    user_prompt = CITATION_QUALITY_USER_PROMPT.render(answer_text=answer_text)
    llm_input = LLMInputPrompt(system_prompt=system_prompt, user_message=user_prompt)
    llm_response = await llm_provider.call_async(llm_input, response_schema=CitationQualitySchema)

    claims: list[ClaimEvaluation] = llm_response.parsed_answer.claims
    citations: list[CitationExcerptEvaluation] = llm_response.parsed_answer.citations
    total_claims = len(claims)
    total_citations = len(citations)
    claims_with_citations = sum(1 for c in claims if c.has_citation)
    citations_with_excerpts = sum(1 for c in citations if c.has_relevant_excerpt)

    return CitationQualityResult(
        claim_evaluations=claims,
        citation_evaluations=citations,
        claims_with_citations_ratio=claims_with_citations / total_claims if total_claims > 0 else 0.0,
        citations_with_excerpts_ratio=citations_with_excerpts / total_citations if total_citations > 0 else 0.0,
    )
