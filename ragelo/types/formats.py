from __future__ import annotations

from enum import Enum
from typing import Any

from ragelo.types.pydantic_models import BaseModel, PydanticBaseModel


class AnswerFormat(str, Enum):
    """Enum that contains the names of the available answer formats"""

    TEXT = "text"
    STRUCTURED = "structured"
    JSON = "json"


class LLMResponseType(BaseModel):
    raw_answer: str
    parsed_answer: float | str | dict[str, Any] | PydanticBaseModel | None
