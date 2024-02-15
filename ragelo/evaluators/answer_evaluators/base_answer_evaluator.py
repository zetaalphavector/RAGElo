"""Base model for dealing with answer evaluators"""

from typing import Callable, Dict, Set, Type

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
from ragelo.types import Query
from ragelo.types.configurations import AnswerEvaluatorConfig


class BaseAnswerEvaluator(BaseEvaluator):
    def __init__(
        self,
        config: AnswerEvaluatorConfig,
        queries: Dict[str, Query],
        answers: Dict[str, Dict[str, str]],
        agents: Set[str],
        llm_provider: BaseLLMProvider,
    ):
        if not queries:
            raise ValueError(
                "You are trying to use an Answer Evaluator without providing queries"
            )
        if not answers:
            raise ValueError(
                "You are trying to use an Answer Evaluator without providing answers"
            )
        self.queries = queries
        self.answers = answers

        if not config.output_file:
            self.output_file = "answers_evaluator.log"
        else:
            self.output_file = config.output_file

        self.agents = agents
        self.llm_provider = llm_provider
        self.config = config

    @classmethod
    def from_config(cls, config: AnswerEvaluatorConfig, llm_provider: BaseLLMProvider):
        queries = cls._load_queries(config.query_path)
        if config.answers_file is None:
            raise ValueError("No answers file provided")
        answers, agents = cls.load_answers_and_agents(config.answers_file, queries)
        return cls(config, queries, answers, agents, llm_provider)

    def __len__(self):
        return len(self.queries)


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
