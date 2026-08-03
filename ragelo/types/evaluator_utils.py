"""Utility functions for evaluator result-type resolution."""

from __future__ import annotations

from typing import Annotated, Literal, get_args, get_origin

from pydantic import BaseModel

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


def default_answer_type(result_type: type[EvaluatorResult]) -> type[BaseModel]:
    """The answer format an evaluator asks the LLM to produce.

    A result's `answer` annotation is the union of formats it must be able to *store*: its own,
    plus those of other judges writing to the same evaluable. The first member is the format the
    evaluator produces, so member order is load-bearing and not merely cosmetic. The union may be
    wrapped in `Annotated` to carry the discriminator.
    """
    annotation = result_type.model_fields["answer"].annotation
    if annotation is None:
        raise ValueError(f"Result type {result_type} does not have an 'answer' field with annotation")
    for candidate in get_args(annotation) or (annotation,):
        if candidate is type(None):
            continue
        if get_origin(candidate) is Annotated:
            candidate = get_args(candidate)[0]
        members = [arg for arg in get_args(candidate) if arg is not type(None)]
        candidate = members[0] if members else candidate
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    raise ValueError(f"Could not determine the answer type for result type {result_type}")
