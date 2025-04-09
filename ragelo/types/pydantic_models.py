from __future__ import annotations

from importlib import metadata

from pydantic import BaseModel as PydanticBaseModel
from typing_extensions import TypeAlias

_PYDANTIC_MAJOR_VERSION: int = int(metadata.version("pydantic").split(".")[0])
ValidationError: type[TypeError] | type[ValueError]
if _PYDANTIC_MAJOR_VERSION == 1:
    from pydantic import root_validator

    validator = root_validator(pre=True)  # type: ignore
    post_validator = root_validator(pre=False)  # type: ignore
    ValidationError = TypeError
    SerializablePydanticBaseModel: TypeAlias = PydanticBaseModel  # type: ignore
else:
    from pydantic import (
        SerializeAsAny,  # type: ignore
        ValidationError,  # type: ignore
        model_validator,  # type: ignore
    )

    validator = model_validator(mode="before")  # type: ignore
    post_validator = model_validator(mode="after")  # type: ignore
    ValidationError = ValidationError
    SerializablePydanticBaseModel: TypeAlias = SerializeAsAny[PydanticBaseModel]  # type: ignore


class BaseModel(PydanticBaseModel):
    @classmethod
    def get_model_fields(cls):
        if _PYDANTIC_MAJOR_VERSION == 1:
            return cls.__fields__  # type: ignore
        else:
            return cls.model_fields  # type: ignore

    def model_dump(self, *args, **kwargs):
        if _PYDANTIC_MAJOR_VERSION == 1:
            return self.dict()  # type: ignore
        else:
            return super().model_dump(*args, **kwargs)  # type: ignore

    @classmethod
    def dump_pydantic(cls, pydantic_model: PydanticBaseModel):
        if _PYDANTIC_MAJOR_VERSION == 1:
            return pydantic_model.dict()  # type: ignore
        else:
            return pydantic_model.model_dump()  # type: ignore
