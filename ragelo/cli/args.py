"""Parse arguments for the cli app"""
import collections.abc
import inspect
import sys
from typing import Any, Callable, Dict, Union, get_args, get_origin, get_type_hints

from typer.models import ArgumentInfo, OptionInfo, ParameterInfo, ParamMeta

from ragelo.types import BaseConfig
from ragelo.types.configurations.base_configs import _PYDANTIC_MAJOR_VERSION

arguments = {
    "queries_file",
    "documents_file",
    "domain_long",
    "answers_file",
    "reasoning_file",
    "evaluations_file",
    "retrieval_evaluator_name",
    "answer_evaluator_name",
}


def get_params_from_function(func: Callable[..., Any]) -> Dict[str, ParamMeta]:
    if sys.version_info >= (3, 10):
        signature = inspect.signature(func, eval_str=True)
    else:
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
                    _type = v.annotation  # type: ignore
                    _outer_type = v.annotation  # type: ignore
                    t_args = get_args(_type)
                else:
                    description = v.field_info.description  # type: ignore
                    _type = v.type_  # type: ignore
                    _outer_type = v.outer_type_  # type: ignore
                    t_args = get_args(_outer_type)
                if not isinstance(v, ParameterInfo):
                    if get_origin(_outer_type) == list:
                        _type = _outer_type
                    # elif _outer_type == Union:
                    # _type = t_args[0]
                    # parse the Union type, defaulting to the str type if possible
                    if len(t_args) > 1:
                        if get_origin(_outer_type) == Union:
                            _type = t_args[0]
                        if (
                            get_origin(_outer_type) == collections.abc.Callable
                            or get_origin(_type) == collections.abc.Callable
                        ):
                            # ignore the callable type and move on.
                            continue
                    if k in arguments:
                        argument = ArgumentInfo(default=v.default, help=description)
                        params[k] = ParamMeta(
                            name=k, default=argument, annotation=_type
                        )
                    else:
                        option = OptionInfo(
                            default=v.default,
                            default_factory=v.default_factory,
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
