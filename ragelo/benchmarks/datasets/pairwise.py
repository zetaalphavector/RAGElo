from __future__ import annotations

from typing import Any

from ragelo import Experiment, get_answer_evaluator
from ragelo.benchmarks.datasets.base import Dataset, T_Outcome
from ragelo.evaluators.answer_evaluators import BaseAnswerEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.evaluables import PairwiseGame
from ragelo.types.formats import LLMUsage
from ragelo.types.results import PairwiseGameEvaluatorResult


def game_usage(result: PairwiseGameEvaluatorResult) -> LLMUsage | None:
    """The tokens of both answer orders, or None when either order has no usage."""
    orders = [order.usage for order in (result.a_vs_b_result, result.b_vs_a_result) if order and order.usage]
    if len(orders) < 2:
        return None
    return LLMUsage(
        input_tokens=sum(usage.input_tokens for usage in orders),
        output_tokens=sum(usage.output_tokens for usage in orders),
        cached_tokens=sum(usage.cached_tokens for usage in orders),
    )


class PairwiseDataset(Dataset[BaseAnswerEvaluator, T_Outcome]):
    """Games between two answers, judged in both answer orders. A variant is the name of a pairwise answer
    evaluator."""

    calls_per_evaluation = 2

    def evaluator_options(self, variant: str) -> dict[str, Any]:
        return {}

    def get_evaluator(self, variant: str, llm_provider: BaseLLMProvider, **kwargs: Any) -> BaseAnswerEvaluator:
        return get_answer_evaluator(variant, llm_provider=llm_provider, **self.evaluator_options(variant), **kwargs)

    def items(self, experiment: Experiment) -> list[PairwiseGame]:
        return [game for query in experiment for game in query.pairwise_games.values()]
