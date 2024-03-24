"""Base model for dealing with answer evaluators"""

from typing import Callable, Dict, Type

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
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


class AnswerEvaluatorFactory:
    registry: Dict[str, Type[BaseAnswerEvaluator]] = {}

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
