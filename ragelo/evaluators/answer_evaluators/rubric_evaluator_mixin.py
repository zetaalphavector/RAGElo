from __future__ import annotations

from typing import Type

from pydantic import BaseModel

from ragelo.generators.rubric_generator import RubricGenerator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.answer_formats import Criterion
from ragelo.types.configurations import RubricEvaluatorConfigBase, RubricGeneratorConfig
from ragelo.types.evaluables import ChatMessage
from ragelo.types.experiment import Experiment
from ragelo.types.query import Query


class RubricEvaluatorMixin:
    """Rubric plumbing shared by the evaluators that grade an answer against `query.rubric`."""

    config: RubricEvaluatorConfigBase
    llm_provider: BaseLLMProvider

    def __init__(self, *args, rubric_generator: RubricGenerator | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_schema_cache: dict[str, Type[BaseModel]] = {}
        self.rubric_generator = rubric_generator or RubricGenerator(
            RubricGeneratorConfig(
                expert_in=self.config.expert_in,
                company=self.config.company,
                n_criteria=self.config.n_criteria,
            ),
            self.llm_provider,
        )

    def _build_evaluation_schema(self, rubric: list[Criterion]) -> Type[BaseModel]:
        raise NotImplementedError

    def _rubric_for(self, query: Query) -> list[Criterion]:
        """The rubric this query is graded against, seeded from `config.rubrics` when it has none."""
        if not query.rubric and self.config.rubrics and query.qid in self.config.rubrics:
            query.rubric = self.config.rubrics[query.qid]
        return query.rubric

    def prepare_experiment(self, experiment: Experiment) -> None:
        contexts = {}
        for query in experiment:
            self._rubric_for(query)
            context = self._rubric_conversation_context(query)
            if context:
                contexts[query.qid] = context
        self.rubric_generator.generate_experiment(experiment, conversation_contexts=contexts)

    def prepare_query(self, query: Query) -> None:
        if not self._rubric_for(query):
            query.rubric = self.rubric_generator.generate(query, self._rubric_conversation_context(query))

    def _rubric_conversation_context(self, query: Query) -> list[ChatMessage]:
        return []

    def _rubric_schema(self, query: Query) -> Type[BaseModel]:
        """The per-query response schema, one field per criterion, memoized per distinct rubric."""
        fingerprint = query.rubric_fingerprint
        if fingerprint is None:
            raise RuntimeError(
                f"Query {query.qid} has no rubric to grade against. Rubrics are produced by "
                "prepare_experiment() / prepare_query(), which evaluate_experiment() and "
                "evaluate_all_evaluables() run for you. Judging a single evaluable directly needs "
                "query.rubric set, config.rubrics, or a RubricGenerator run first."
            )
        if fingerprint not in self.answer_schema_cache:
            self.answer_schema_cache[fingerprint] = self._build_evaluation_schema(query.rubric)
        return self.answer_schema_cache[fingerprint]
