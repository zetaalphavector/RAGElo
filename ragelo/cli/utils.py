from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider, split_llm_provider_kwargs


def get_path(data_path: str | None, file_path: str, check_exists: bool = True) -> str:
    if data_path is None:
        data_path = "."
    if file_path.startswith("/"):
        abs_path = file_path
    else:
        abs_path = os.path.abspath(os.path.join(data_path, file_path))
    if check_exists:
        assert os.path.exists(abs_path), f"File {abs_path} does not exist"
    return abs_path


def config_kwargs(config_class: type[BaseModel], cli_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cli_kwargs.items() if k in config_class.model_fields}


def get_cli_llm_provider(name: str, cli_kwargs: dict[str, Any]) -> BaseLLMProvider:
    provider_kwargs, _ = split_llm_provider_kwargs(name, cli_kwargs)
    return get_llm_provider(name, **{option: value for option, value in provider_kwargs.items() if value is not None})
