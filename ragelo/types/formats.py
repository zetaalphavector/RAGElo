from __future__ import annotations

from enum import Enum


class AnswerFormat(str, Enum):
    """Enum that contains the names of the available answer formats"""

    JSON = "json"
    TEXT = "text"
    MULTI_FIELD_JSON = "multi_field_json"
