"""Typer commands that take one config object, with one CLI parameter per field of that config."""

from __future__ import annotations

import collections.abc
import functools
import inspect
from collections.abc import Callable
from types import NoneType, UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from jinja2 import Template
from pydantic.fields import FieldInfo
from typer.models import ArgumentInfo, OptionInfo

from ragelo.types import BaseConfig


def cli_type(field: FieldInfo) -> Any | None:
    """The type Typer parses the field as. None for a field the command line cannot express."""
    parsed = field.annotation
    if get_origin(parsed) in (Union, UnionType):
        parsed = next(choice for choice in get_args(parsed) if choice is not NoneType)
    if parsed in (NoneType, Template) or get_origin(parsed) in (dict, type, collections.abc.Callable):
        return None
    return parsed


def config_parameters(config_class: type[BaseConfig]) -> list[inspect.Parameter]:
    parameters = []
    for name, field in config_class.model_fields.items():
        parsed = cli_type(field)
        if parsed is None:
            continue
        if any(isinstance(marker, ArgumentInfo) for marker in field.metadata):
            default: Any = ArgumentInfo(default=field.default, help=field.description)
        else:
            default = OptionInfo(
                default=field.default,
                default_factory=field.default_factory,  # type: ignore[arg-type]
                help=field.description,
            )
        parameters.append(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default, annotation=parsed))
    return parameters


def config_command(func: Callable[[Any], Any]) -> Callable[..., Any]:
    """Typer reads a command's parameters from its signature, so the command it registers takes the fields
    of the config, and `func` takes the config built from them."""
    [config_name] = inspect.signature(func).parameters
    config_class = get_type_hints(func)[config_name]
    parameters = config_parameters(config_class)

    @functools.wraps(func)
    def command(**kwargs: Any) -> Any:
        return func(config_class(**kwargs))

    command.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    command.__annotations__ = {parameter.name: parameter.annotation for parameter in parameters}
    return command
