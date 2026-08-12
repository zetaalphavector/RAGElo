"""Utility functions for evaluator result-type resolution."""

from __future__ import annotations

import warnings
from typing import Annotated, Any, Literal, get_args, get_origin

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
    """The first answer format a result type can store, optionally through an `Annotated` union.

    Only `answer_format_for` calls this, to keep evaluators working that swap `result_type` without
    declaring the format they produce. Prefer declaring `answer_format`: a result class is shared
    across judges, so which member comes first in its union is an ordering accident.
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


def answer_format_for(evaluator: Any, base: Any) -> type[BaseModel]:
    """The answer format an evaluator asks the LLM to produce.

    Evaluators declare it as `answer_format`. Deriving it from `result_type` instead is deprecated:
    a result class is shared by every judge writing to the same evaluable, so its `answer` union
    lists all the formats it can *store*, and using that union's first member as the request schema
    lets a reordering of the union silently change what the LLM is asked for.

    `base` is the evaluator base class whose `answer_format` counts as the inherited default; a
    declaration on the evaluator itself, or on any class between it and `base`, is authoritative.
    The deprecated derivation is reached only by an evaluator that swaps `result_type` for a custom
    one without declaring the format it produces, which is how the two used to be kept in step.
    """
    declared = evaluator.answer_format
    if "answer_format" in vars(evaluator) or _declares_answer_format(type(evaluator), base):
        return declared
    if evaluator.result_type is base.result_type:
        return declared
    derived = default_answer_type(evaluator.result_type)
    if derived is declared:
        return declared
    warnings.warn(
        f"{type(evaluator).__name__} overrides result_type without declaring answer_format, so the "
        f"schema requested from the LLM is being derived from {evaluator.result_type.__name__}. "
        f"Declare `answer_format = {derived.__name__}` on the evaluator instead; deriving it is "
        "deprecated because it makes the result type's union order significant.",
        DeprecationWarning,
        stacklevel=3,
    )
    return derived


def _declares_answer_format(cls: type, base: Any) -> bool:
    for klass in cls.__mro__:
        if klass is base:
            return False
        if "answer_format" in vars(klass):
            return True
    return False
