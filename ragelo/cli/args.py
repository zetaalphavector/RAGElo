"""Parse arguments for the cli app"""

import inspect
from typing import Any, Callable, get_type_hints

from typer.models import ArgumentInfo, OptionInfo, ParameterInfo, ParamMeta

from ragelo.types import BaseConfig
from ragelo.types.configurations.base_configs import _PYDANTIC_MAJOR_VERSION

arguments = {
    "query_path",
    "documents_path",
    "domain_long",
    "answers_file",
    "reasoning_file",
    "evaluations_file",
    "retrieval_evaluator_name",
    "answer_evaluator_name",
    "output_file",
}


def get_params_from_function(func: Callable[..., Any]) -> dict[str, ParamMeta]:
    signature = inspect.signature(func)

    type_hints = get_type_hints(func)

    params = {}

    for param in signature.parameters.values():
        annotation = param.annotation
        if param.name == "kwargs" or param.name == "args":
            continue
        if param.name in type_hints:
            annotation = type_hints[param.name]
        if inspect.isclass(annotation) and issubclass(annotation, BaseConfig):
            fields = annotation.get_model_fields()
            for k, v in fields.items():
                if _PYDANTIC_MAJOR_VERSION == 2:
                    description = v.description  # type: ignore
                    _type = v.annotation
                else:
                    description = v.field_info.description  # type: ignore
                    _type = v.type_
                if not isinstance(v, ParameterInfo):
                    # get the description from Pydantic model
                    if k in arguments:
                        argument = ArgumentInfo(default=v.default, help=description)
                        params[k] = ParamMeta(
                            name=k, default=argument, annotation=_type
                        )
                    else:
                        option = OptionInfo(
                            default=v.default,
                            help=description,
                        )
                        params[k] = ParamMeta(name=k, default=option, annotation=_type)
                else:
                    params[k] = ParamMeta(name=k, default=v, annotation=_type)

        else:
            params[param.name] = ParamMeta(
                name=param.name, default=param.default, annotation=annotation
            )
    return params
