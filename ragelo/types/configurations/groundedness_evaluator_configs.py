from typing import Any, Optional, Type

from pydantic import BaseModel, Field
from ragelo.types.answer_formats import GroundednessEvaluatorFormat
from ragelo.types.configurations.base_configs import BaseEvaluatorConfig


class BaseGroundednessEvaluatorConfig(BaseEvaluatorConfig):
    """
    A base configuration class for groundedness evaluators.
    """

    llm_response_schema: Optional[Type[BaseModel] | dict[str, Any]] = Field(
        default=GroundednessEvaluatorFormat
    )
