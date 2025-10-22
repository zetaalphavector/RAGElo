"""Utility functions for evaluators that avoid circular imports."""

from __future__ import annotations

from ragelo.types.results import EvaluatorResult
from ragelo.types.types import AnswerEvaluatorTypes, RetrievalEvaluatorTypes


def get_evaluator_result_type(
    evaluator_name: str | AnswerEvaluatorTypes | RetrievalEvaluatorTypes,
) -> type[EvaluatorResult]:
    """Gets the result type for any evaluator (answer or retrieval).

    Args:
        evaluator_name: The name of the evaluator (string or enum).

    Returns:
        type[EvaluatorResult]: The result type for the evaluator.

    Raises:
        ValueError: If the evaluator name is not found in either registry.
    """
    # Import here to avoid circular imports
    from ragelo.evaluators.answer_evaluators.base_answer_evaluator import AnswerEvaluatorFactory
    from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import RetrievalEvaluatorFactory

    # Convert string to appropriate enum if needed
    evaluator_name_str = str(evaluator_name)

    # Try AnswerEvaluatorTypes first
    try:
        answer_enum = AnswerEvaluatorTypes(evaluator_name_str)
        return AnswerEvaluatorFactory.get_evaluator_result_type(answer_enum)
    except (ValueError, KeyError):
        pass

    # Try RetrievalEvaluatorTypes
    try:
        retrieval_enum = RetrievalEvaluatorTypes(evaluator_name_str)
        return RetrievalEvaluatorFactory.get_evaluator_result_type(retrieval_enum)
    except (ValueError, KeyError):
        pass

    # If neither worked, raise an error
    raise ValueError(
        f"Unknown evaluator '{evaluator_name}'. "
        f"Valid answer evaluators: {[e.value for e in AnswerEvaluatorTypes]}. "
        f"Valid retrieval evaluators: {[e.value for e in RetrievalEvaluatorTypes]}."
    )
