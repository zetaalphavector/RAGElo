"""Base model for dealing with answer evaluators"""

import csv
import logging
import os
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any, Callable, List, Optional, Type

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
from ragelo.types import AgentAnswer, Query
from ragelo.types.configurations import AnswerEvaluatorConfig


class BaseAnswerEvaluator(BaseEvaluator):
    def __init__(
        self,
        config: AnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        if not config.output_file:
            self.output_file = "answers_evaluator.log"
        else:
            self.output_file = config.output_file

        self.llm_provider = llm_provider
        self.config = config

    @classmethod
    def from_config(cls, config: AnswerEvaluatorConfig, llm_provider: BaseLLMProvider):
        return cls(config, llm_provider)

    @abstractmethod
    def run(
        self, queries: List[Query], answers: List[AgentAnswer]
    ) -> List[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_single_sample(
        self, answer: AgentAnswer | Sequence[AgentAnswer]
    ) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def _process_answer(self, answer: str) -> Any:
        """Processes the LLM evaluator output into some serializable format"""
        raise NotImplementedError

    def _dump_response(self, answer_dict: dict[str, str], file: Optional[str] = None):
        output_file = file if file else self.output_file
        if not os.path.isfile(output_file):
            logging.debug(f"Creating new file {output_file}")
            with open(output_file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=self.output_columns)
                writer.writeheader()
        self._print_response(answer_dict)
        with open(output_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.output_columns)


class AnswerEvaluatorFactory:
    registry: dict[str, Type[BaseAnswerEvaluator]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Type[BaseAnswerEvaluator]):
            if name in cls.registry:
                logger.warning(f"Overwriting {name} in registry")
            cls.registry[name.lower()] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        evaluator_name: str,
        config: AnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        if evaluator_name.lower() not in cls.registry:
            raise ValueError(f"Unknown evaluator {evaluator_name}")
        return cls.registry[evaluator_name.lower()].from_config(config, llm_provider)
