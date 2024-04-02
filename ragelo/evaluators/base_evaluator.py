import json
from abc import ABC, abstractmethod

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
