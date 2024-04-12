"""Parse arguments for the cli app"""

import inspect
from typing import Any, Callable, get_type_hints

import pydantic
from typer.models import ArgumentInfo, OptionInfo, ParameterInfo, ParamMeta

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
        if inspect.isclass(annotation) and issubclass(annotation, pydantic.BaseModel):
            fields = annotation.model_fields
            for k, v in fields.items():
                # k, v) in enumerate(dct.items()):
                _type = v.type_
                if not isinstance(v, ParameterInfo):
                    # get the description from Pydantic model
                    description = v.field_info.description

                    if k in arguments:
                        v = ArgumentInfo(default=v.default, help=description)
                    else:
                        v = OptionInfo(
                            default=v.default,
                            help=description,
                        )

                params[k] = ParamMeta(name=k, default=v, annotation=_type)
        else:
            params[param.name] = ParamMeta(
                name=param.name, default=param.default, annotation=annotation
            )
    return params
