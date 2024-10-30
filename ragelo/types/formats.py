from __future__ import annotations

from enum import Enum


class AnswerFormat(str, Enum):
    """Enum that contains the names of the available answer formats"""

    TEXT = "text"
    STRUCTURED = "structured"
    JSON = "json"
