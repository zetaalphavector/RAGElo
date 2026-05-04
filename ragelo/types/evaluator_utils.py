"""Utility functions for evaluator result-type resolution."""

from __future__ import annotations

from typing import Literal

from ragelo.types.evaluables import AgentAnswer, Document, PairwiseGame
from ragelo.types.results import EvaluatorResult
from ragelo.types.types import AnswerEvaluatorTypes, RetrievalEvaluatorTypes, _result_type_registry


def resolve_evaluator_result_type(
    evaluator_name: str | AnswerEvaluatorTypes | RetrievalEvaluatorTypes,
    evaluable: object | None = None,
    kind: Literal["retrieval", "answer"] | None = None,
) -> type[EvaluatorResult]:
    """Resolve the expected result type for an evaluator.

    - If `kind` is provided, route directly to the matching registry. This is the
      most reliable disambiguation when the evaluable object is not available
      (e.g., when reloading persisted results from disk).
    - If `evaluable` is provided, route to the proper registry by evaluable kind to
      avoid name collisions (e.g., domain_expert existing for both answer and retrieval).
    - Otherwise, fall back to name-only resolution (answer first, then retrieval).
    """
    name_str = str(evaluator_name)

    # Explicit kind takes precedence.
    if kind == "retrieval":
        try:
            return _result_type_registry[f"retrieval:{RetrievalEvaluatorTypes(name_str)}"]
        except (ValueError, KeyError):
            raise ValueError(f"Unknown retrieval evaluator: {name_str}")
    if kind == "answer":
        try:
            return _result_type_registry[f"answer:{AnswerEvaluatorTypes(name_str)}"]
        except (ValueError, KeyError):
            raise ValueError(f"Unknown answer evaluator: {name_str}")

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
