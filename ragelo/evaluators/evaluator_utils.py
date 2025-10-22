"""Utility functions for evaluator result-type resolution."""

from __future__ import annotations

from ragelo.types.results import EvaluatorResult
from ragelo.types.types import AnswerEvaluatorTypes, RetrievalEvaluatorTypes


def resolve_evaluator_result_type(
    evaluator_name: str | AnswerEvaluatorTypes | RetrievalEvaluatorTypes,
    evaluable: object | None = None,
) -> type[EvaluatorResult]:
    """Resolve the expected result type for an evaluator.

    - If `evaluable` is provided, route to the proper registry by evaluable kind to
      avoid name collisions (e.g., domain_expert existing for both answer and retrieval).
    - If `evaluable` is None, fall back to name-only resolution (answer first, then retrieval).
    """
    # Import locally to avoid circular imports
    from ragelo.evaluators.answer_evaluators.base_answer_evaluator import AnswerEvaluatorFactory
    from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import RetrievalEvaluatorFactory
    from ragelo.types.evaluables import AgentAnswer, Document, PairwiseGame

    name_str = str(evaluator_name)

    # Contextual resolution
    if evaluable is not None:
        if isinstance(evaluable, Document):
            try:
                return RetrievalEvaluatorFactory.get_evaluator_result_type(RetrievalEvaluatorTypes(name_str))
            except Exception:
                pass
        if isinstance(evaluable, (AgentAnswer, PairwiseGame)):
            try:
                return AnswerEvaluatorFactory.get_evaluator_result_type(AnswerEvaluatorTypes(name_str))
            except Exception:
                pass

    # Name-only resolution (Answer first, then Retrieval)
    try:
        return AnswerEvaluatorFactory.get_evaluator_result_type(AnswerEvaluatorTypes(name_str))
    except Exception:
        pass
    return RetrievalEvaluatorFactory.get_evaluator_result_type(RetrievalEvaluatorTypes(name_str))
