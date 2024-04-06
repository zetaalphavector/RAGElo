import json
import string
from abc import ABC, abstractmethod
from typing import Optional

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import BaseEvaluatorConfig


class BaseEvaluator(ABC):
    @abstractmethod
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        config: BaseEvaluatorConfig,
    ):
        raise NotImplementedError

    @staticmethod
    def json_answer_parser(answer: str, key: str) -> str:
        """Parses a Json answer from the LLM and returns a specific key"""

        # Finds all valid JSON objects in the answer that contain the key
        json_objects = []
        for line in answer.strip().split("\n"):
            try:
                json_object = json.loads(line)
                if key in json_object:
                    json_objects.append(json_object)
            except json.JSONDecodeError:
                pass

        # Assumes the valid JSON object is the last one
        if not json_objects:
            raise ValueError(
                "Answer does not contain a valid json object\n"
                f"with the key {key}\n{answer}"
            )
        json_dict = json_objects[-1]
        return json_dict[key]

    @staticmethod
    def json_answer_parser_multifields(answer: str, keys: list[str]) -> dict[str, str]:
        """Parses a Json answer from the LLM and returns the values from multiple fields"""

        # Finds all valid JSON objects in the answer that contain the key
        values = {}
        for line in answer.strip().split("\n"):
            try:
                json_object = json.loads(line)
                for k in keys:
                    if k in json_object:
                        values[k] = json_object[k]
            except json.JSONDecodeError:
                pass

        if len(values) != len(keys):
            raise ValueError(
                "Answer does not contain all necessary keys\n"
                f"Expected {keys}, found {values.keys()}.\n"
                f"Full Answer:\n{answer}"
            )
        # Assumes the valid JSON object is the last one
        return values

    @staticmethod
    def _get_fields_from_string(s: str) -> list[str]:
        """Parse a formatted string and return all the fields in it"""
        field_names = [v[1] for v in string.Formatter().parse(s) if v[1] is not None]
        return field_names

    @staticmethod
    def _get_valid_fields_from_metadata(
        prompt: str, metadata: Optional[dict[str, str]], skip_fields: list[str] = []
    ) -> dict[str, str]:
        """Get the fields from the prompt that are in the metadata"""
        expected_fields = BaseEvaluator._get_fields_from_string(prompt)
        valid_fields = {}
        if metadata is None:
            return valid_fields
        for field in expected_fields:
            if field in metadata and field not in skip_fields:
                valid_fields[field] = metadata[field]
        return valid_fields
