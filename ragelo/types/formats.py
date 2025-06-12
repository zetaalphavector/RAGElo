from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel
from typing_extensions import TypeAlias

LLMResponseType: TypeAlias = str | dict[str, Any] | BaseModel
ResponseSchema: TypeAlias = Type[BaseModel] | dict[str, Any]
