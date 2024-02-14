"""Base model for dealing with answer evaluators"""

import csv
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, Optional, Set, Type

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
from ragelo.types import Query
from ragelo.types.configurations import AnswerEvaluatorConfig


class AnswerEvaluator(BaseEvaluator):
    def __init__(
        self,
        config: AnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
        queries: Optional[Dict[str, Query]] = None,
        answers: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        if not queries and not config.query_path:
            raise ValueError(
                "You are trying to use an Answer Evaluator without providing queries"
            )
        if not answers and not config.answers_file:
            raise ValueError(
                "You are trying to use an Answer Evaluator without providing answers"
            )
        self.queries = self.load_queries(config.query_path) if not queries else queries
        self.answers = (
            self._load_answers(config.answers_file) if not answers else answers
        )

        if not self.config.output_file:
            self.output_file = f"answers_evaluator.log"
        else:
            self.output_file = self.config.output_file
        self.agents: Set[str] = set()
        self.llm_provider = llm_provider

    def _load_answers(self, answers_path: str) -> Dict[str, Dict[str, str]]:
        answers: Dict[str, Dict[str, str]] = defaultdict(lambda: dict())
        for line in csv.DictReader(open(answers_path)):
            qid = line["query_id"]
            if qid not in self.queries:
                continue
            agent = line["agent"]
            self.agents.add(agent)
            answer = line["answer"]
            answers[qid][agent] = answer
        self._check_validity()
        return answers

    @abstractmethod
    def _check_validity(self):
        raise NotImplementedError


class AnswerEvaluatorFactory:
    registry: Dict[str, Type[AnswerEvaluator]] = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type[AnswerEvaluator]):
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
        queries: Optional[Dict[str, Query]] = None,
        answers: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        if evaluator_name not in cls.registry:
            raise ValueError(f"Unknown evaluator {evaluator_name}")
        return cls.registry[evaluator_name.lower()](config, queries, answers)
