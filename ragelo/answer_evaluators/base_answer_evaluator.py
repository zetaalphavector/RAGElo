"""Base model for dealing with answer evaluators"""
import csv
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, Set, Type

from ragelo.logger import logger
from ragelo.utils.openai_client import OpenAiClient, set_credentials_from_file


class AnswerEvaluator:
    def __init__(
        self,
        query_path: str,
        answers_file: str,
        output_file: str,
        evaluator_name: str,
        model_name: str = "gpt-4",
        credentials_file: str | None = None,
        print_answers: bool = False,
        force: bool = False,
        **kwargs,
    ):
        self.name = evaluator_name
        self.print = print_answers
        self.force = force
        self.agents: Set[str] = set()
        self.output_file = output_file
        self.queries = self._load_queries(query_path)
        self.answers = self._load_answers(answers_file)

        if credentials_file:
            set_credentials_from_file(credentials_file)

        self.openai_client = OpenAiClient(model=model_name)

    @abstractmethod
    def run(self):
        """Run and extract answers for all queries"""
        pass

    def _load_queries(self, queries_path: str) -> Dict[str, str]:
        queries = {}
        if not os.path.isfile(queries_path):
            logger.exception(f"Queries file {queries_path} not found")
            raise FileNotFoundError

        for line in csv.reader(open(queries_path)):
            if "query" in line:
                continue
            qid, query = line
            queries[qid] = query
        if self.print:
            logger.info(f"Loaded {len(queries)} queries")
        return queries

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
    def create(cls, name: str, **kwargs) -> AnswerEvaluator:
        if name.lower() not in cls.registry:
            raise ValueError(f"Name {name} not in registry")
        return cls.registry[name.lower()](evaluator_name=name, **kwargs)
