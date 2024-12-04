"""Parse arguments for the cli app"""

from __future__ import annotations

import collections.abc
import inspect
import sys
from typing import Any, Callable, Dict, get_args, get_origin, get_type_hints

from typer.models import ArgumentInfo, OptionInfo, ParameterInfo, ParamMeta

from ragelo.types import BaseConfig
from ragelo.types.pydantic_models import _PYDANTIC_MAJOR_VERSION

arguments = {
    "queries_csv_file",
    "documents_csv_file",
    "answers_csv_file",
    "domain_long",
}

ignore_args = {"llm_response_schema"}


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
                if k in ignore_args:
                    continue
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
                if get_origin(_outer_type) is list:
                    _type = _outer_type
                if get_origin(_outer_type) is type(None) or _type is type(None):
                    continue
                if get_origin(_outer_type) is dict:
                    continue

                if not isinstance(v, ParameterInfo):
                    if len(t_args) > 1:
                        # To resolve the True argument type, first remove any NoneType from the list of types"
                        _t_args = [t for t in t_args if t is not type(None)]
                        if len(_t_args) == 1:
                            _type = _t_args[0]
                        if (
                            get_origin(_outer_type) == collections.abc.Callable
                            or get_origin(_type) == collections.abc.Callable
                        ):
                            # ignore the callable type and move on.
                            continue
                        elif len(_t_args) > 1:
                            _type = _t_args[0]
                    if k in arguments:
                        argument = ArgumentInfo(default=v.default, help=description)
                        params[k] = ParamMeta(name=k, default=argument, annotation=_type)
                    else:
                        option = OptionInfo(
                            default=v.default,
                            default_factory=v.default_factory,  # type: ignore
                            help=description,
                        )
                        params[k] = ParamMeta(name=k, default=option, annotation=_type)
                else:
                    params[k] = ParamMeta(name=k, default=v, annotation=_type)

        else:
            params[param.name] = ParamMeta(name=param.name, default=param.default, annotation=annotation)
    return params
