from __future__ import annotations

from enum import Enum
from typing import Any

from typing_extensions import TypeAlias

from ragelo.types.pydantic_models import BaseModel


class AnswerFormat(str, Enum):
    """Enum that contains the names of the available answer formats"""

    TEXT = "text"
    STRUCTURED = "structured"
    JSON = "json"


LLMResponseType: TypeAlias = BaseModel | str | dict[str, Any]
