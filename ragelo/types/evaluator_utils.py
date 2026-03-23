"""Utility functions for evaluator result-type resolution."""

from __future__ import annotations

from ragelo.types.evaluables import AgentAnswer, Document, PairwiseGame
from ragelo.types.results import EvaluatorResult
from ragelo.types.types import AnswerEvaluatorTypes, RetrievalEvaluatorTypes, _result_type_registry


def resolve_evaluator_result_type(
    evaluator_name: str | AnswerEvaluatorTypes | RetrievalEvaluatorTypes,
    evaluable: object | None = None,
) -> type[EvaluatorResult]:
    """Resolve the expected result type for an evaluator.

    - If `evaluable` is provided, route to the proper registry by evaluable kind to
      avoid name collisions (e.g., domain_expert existing for both answer and retrieval).
    - If `evaluable` is None, fall back to name-only resolution (answer first, then retrieval).
    """
    name_str = str(evaluator_name)

    # Contextual resolution
    if evaluable is not None:
        if isinstance(evaluable, Document):
            try:
                return _result_type_registry[f"retrieval:{RetrievalEvaluatorTypes(name_str)}"]
            except (ValueError, KeyError):
                pass
        if isinstance(evaluable, (AgentAnswer, PairwiseGame)):
            try:
                return _result_type_registry[f"answer:{AnswerEvaluatorTypes(name_str)}"]
            except (ValueError, KeyError):
                pass

    # Name-only resolution (Answer first, then Retrieval)
    try:
        return _result_type_registry[f"answer:{AnswerEvaluatorTypes(name_str)}"]
    except (ValueError, KeyError):
        pass
    try:
        return _result_type_registry[f"retrieval:{RetrievalEvaluatorTypes(name_str)}"]
    except (ValueError, KeyError):
        raise ValueError(f"Unknown evaluator: {name_str}")
