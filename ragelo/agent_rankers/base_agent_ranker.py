"""Base models for ranking a set of answers generated by LLMs"""

import csv
import json
from abc import abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple, Type

from ragelo.logger import logger
from ragelo.types import AgentRankerConfig


class AgentRanker:
    config: AgentRankerConfig
    evaluations: List[Tuple[str, str, str]]
    ranking: defaultdict

    def __init__(
        self,
        config: AgentRankerConfig,
        evaluations: List[Tuple[str, str, str]],
    ):
        self.config = config
        self.evaluations = evaluations
        self.ranking = defaultdict(list)
        self.output_file = config.output_file

    @staticmethod
    def load_evaluations(answers_file: str) -> List[Tuple[str, str, str]]:
        evaluations = []
        for line in open(answers_file, "r"):
            data = json.loads(line)
            agent_a = data["agent_a"]
            agent_b = data["agent_b"]
            relevant = data["relevant"]
            evaluations.append((agent_a, agent_b, relevant))
        return evaluations

    @classmethod
    def from_config(cls, config: AgentRankerConfig):
        evaluations = cls.load_evaluations(config.evaluations_file)
        return cls(config, evaluations)

    @abstractmethod
    def run(self):
        """Compute score for each agent"""
        raise NotImplementedError

    @abstractmethod
    def get_agents_ratings(self) -> Dict[str, float]:
        """Returns the score of all players"""
        raise NotImplementedError

    def dump_ranking(self):
        with open(self.output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["agent", "score"])
            for agent, rating in sorted(
                self.get_agents_ratings().items(), key=lambda x: x[1], reverse=True
            ):
                writer.writerow([agent, rating])

    def print_ranking(self):
        if not self.config.verbose:
            return
        scores = sorted(
            self.get_agents_ratings().items(), key=lambda x: x[1], reverse=True
        )
        if self.config.rich_print:
            try:
                import rich

                rich.print(
                    f"-------[bold white] Agent Scores by {self.name} [/bold white]-------"
                )

                for agent, rating in scores:
                    rich.print(f"[bold white]{agent:<15}[/bold white]: {rating:.1f}")
            except ImportError:
                logger.warning("Rich not installed. Using plain print")
                self.config.rich_print = False
        if not self.config.rich_print:
            print(f"------- Agent Scores by {self.name} -------")
            for agent, rating in scores:
                print(f"{agent:<15}: {rating:.1f}")


class AgentRankerFactory:
    registry: Dict[str, Type[AgentRanker]] = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type[AgentRanker]):
            if name in cls.registry:
                logger.warning(f"Overwriting {name} in Answer Evaluator registry")
            cls.registry[name.lower()] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, ranker_name: str, config: AgentRankerConfig) -> AgentRanker:
        if ranker_name.lower() not in cls.registry:
            raise ValueError(f"Unknown Agent Ranker {ranker_name}")
        return cls.registry[ranker_name.lower()].from_config(config)