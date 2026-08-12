"""Evaluator that grades a document by which of a query's rubric criteria it addresses"""

from __future__ import annotations

from pydantic import BaseModel, Field, create_model

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.evaluators.rubric_evaluator_mixin import RubricEvaluatorMixin
from ragelo.types.answer_formats import Criterion, RubricCoverageAnswerFormat
from ragelo.types.configurations import RubricCoverageEvaluatorConfig
from ragelo.types.evaluables import Document
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.query import Query
from ragelo.types.types import RetrievalEvaluatorTypes
from ragelo.utils import string_to_template


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.RUBRIC_COVERAGE)
class RubricCoverageEvaluator(RubricEvaluatorMixin, BaseRetrievalEvaluator[RubricCoverageEvaluatorConfig]):
    """A retrieval evaluator that asks, per criterion, whether a document addresses it.

    Where a relevance evaluator asks for one holistic score, this decomposes the judgement into
    the query's rubric (`query.rubric`) and asks the LLM to verify each criterion separately, so
    the judgement is inspectable criterion by criterion. The rubric is an artifact settled before
    judging starts: supplied on the query, seeded from `config.rubrics`, or generated in the
    prepare phase from `config.rubric_source` (the pooled documents or the reference answer).

    The result carries the addressed criteria, letting the same judgement feed both flat qrels
    (`score` = how many criteria the document addresses) and subtopic qrels for coverage
    measures such as `StRecall` and `alpha_nDCG`.
    """

    config: RubricCoverageEvaluatorConfig
    answer_format = RubricCoverageAnswerFormat

    system_prompt = string_to_template("""
        You are an impartial expert document annotator and a domain expert in {{ expert_in }}.{% if company %} You are annotating for {{ company }}.{% endif %}
        You are tasked with evaluating a retrieval system for question answering.

        A user asked the question you will be shown. You are given a rubric: the criteria that a
        complete answer to that question must satisfy. For each criterion, decide whether the
        retrieved document addresses it.

        Judge the document, not the question, and judge each criterion on its own:
        - A criterion is addressed only if the document states the information the criterion asks for.
        - Sharing a system name, a process name or vocabulary with the question does not address a
          criterion if the document does not carry the information itself.
        - A document may address a criterion in different words than the criterion uses, and may fail
          to address one it superficially resembles.
        If you are uncertain about a criterion, treat it as not addressed.

        ## Rubric
        {% for criterion in rubric %}
        Criterion: {{ criterion.criterion_name }}
        Question: {{ criterion.short_question }}
        --------------------------------
        {% endfor %}
        """)  # noqa: E501

    user_prompt = string_to_template("""
        [User Question]
        {{ query.query }}

        [Retrieved Document]
        {% if document.metadata and document.metadata.title %}Title: {{ document.metadata.title }}
        {% endif %}{{ document.text }}
        """)

    def _build_evaluation_schema(self, rubric: list[Criterion]) -> type[BaseModel]:
        criteria_models = {
            criterion.criterion_name: (
                bool,
                Field(description=f"Whether the document addresses this criterion: {criterion.short_question}"),
            )
            for criterion in rubric
        }
        return create_model(
            "RubricCoverageSchema",
            reasoning=(
                str,
                Field(description="A concise explanation of which criteria the document addresses, and why."),
            ),
            **criteria_models,  # type: ignore[call-overload]
        )

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        schema = self._rubric_schema(query)
        return LLMInputPrompt(
            system_prompt=self.system_prompt.render(
                expert_in=self.config.expert_in,
                company=self.config.company,
                rubric=query.rubric,
            ),
            user_message=self.user_prompt.render(query=query, document=document),
            llm_response_schema=schema,
        )

    def _process_answer(self, llm_response: LLMResponseType, query: Query) -> LLMResponseType:
        response = llm_response.parsed_answer.model_dump()
        addressed = [c.criterion_name for c in query.rubric if response.get(c.criterion_name)]
        return LLMResponseType(
            raw_answer=llm_response.raw_answer,
            parsed_answer=RubricCoverageAnswerFormat(
                reasoning=response.get("reasoning", ""),
                criteria_addressed=addressed,
                score=len(addressed),
                rubric_fingerprint=query.rubric_fingerprint,
            ),
        )
