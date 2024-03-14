"""Parse arguments for the cli app"""

import dataclasses
import inspect
from dataclasses import fields
from typing import Any, Callable, Dict, get_type_hints

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
}


def get_params_from_function(func: Callable[..., Any]) -> Dict[str, ParamMeta]:
    signature = inspect.signature(func)

    type_hints = get_type_hints(func)

    params = {}

    for param in signature.parameters.values():
        annotation = param.annotation
        if param.name == "kwargs" or param.name == "args":
            continue
        if param.name in type_hints:
            annotation = type_hints[param.name]
        if inspect.isclass(annotation) and dataclasses.is_dataclass(annotation):
            dct = dataclasses.asdict(annotation())
            subtype_hints = get_type_hints(annotation)
            for idx, (k, v) in enumerate(dct.items()):
                if not isinstance(v, ParameterInfo):
                    help = fields(param.default)[idx].metadata.get("help", "")
                    if k in arguments:
                        v = ArgumentInfo(default=v, help=help)
                    else:
                        v = OptionInfo(
                            default=v,
                            help=help,
                        )

                params[k] = ParamMeta(
                    name=k, default=v, annotation=subtype_hints.get(k, str)
                )
        else:
            params[param.name] = ParamMeta(
                name=param.name, default=param.default, annotation=annotation
            )
    return params
