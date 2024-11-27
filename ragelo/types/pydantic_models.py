from __future__ import annotations

from importlib import metadata

from pydantic import BaseModel as PydanticBaseModel

_PYDANTIC_MAJOR_VERSION: int = int(metadata.version("pydantic").split(".")[0])
if _PYDANTIC_MAJOR_VERSION == 1:
    from pydantic import root_validator

    validator = root_validator(pre=True)  # type: ignore
    post_validator = root_validator(pre=False)  # type: ignore
    ValidationError = TypeError
else:
    from pydantic import (
        ValidationError,  # type: ignore
        model_validator,  # type: ignore
    )

    validator = model_validator(mode="before")  # type: ignore
    post_validator = model_validator(mode="after")  # type: ignore
    ValidationError = ValidationError


class BaseModel(PydanticBaseModel):
    @classmethod
    def get_model_fields(cls):
        if _PYDANTIC_MAJOR_VERSION == 1:
            return cls.__fields__  # type: ignore
        else:
            return cls.model_fields  # type: ignore

    def model_dump(self):
        if _PYDANTIC_MAJOR_VERSION == 1:
            return self.dict()  # type: ignore
        else:
            return super().model_dump()  # type: ignore
