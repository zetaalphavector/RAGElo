from __future__ import annotations

import csv
from typing import Type, get_type_hints

from ragelo.logger import logger
from ragelo.types.configurations.agent_ranker_configs import AgentRankerConfig
from ragelo.types.queries import Queries
from ragelo.types.types import AgentRankerTypes


class AgentRanker:
    config: AgentRankerConfig

    def __init__(
        self,
        config: AgentRankerConfig,
    ):
        self.config = config
        self.name = self.config.ranker_name
        self.agents_evaluations_file = self.config.agents_evaluations_file

    def run(self, queries: Queries) -> dict[str, int]:
        """Compute score for each agent"""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: AgentRankerConfig):
        return cls(config)

    def get_agents_ratings(self) -> dict[str, float]:
        """Returns the score of all players"""
        raise NotImplementedError

    @classmethod
    def get_config_class(cls) -> Type[AgentRankerConfig]:
        return get_type_hints(cls)["config"]

    def _flatten_evaluations(self, queries: Queries) -> list[tuple[str, str, str]]:
        evaluations = []
        for q in queries:
            for game in q.pairwise_games:
                evaluations.append(
                    (
                        game.agent_a_answer.agent,
                        game.agent_b_answer.agent,
                        game.evaluation.answer,
                    )
                )
        return evaluations


class AgentRankerFactory:
    registry: dict[AgentRankerTypes, Type[AgentRanker]] = {}

    @classmethod
    def register(cls, name: AgentRankerTypes):
        def inner_wrapper(wrapped_class: Type[AgentRanker]):
            if name in cls.registry:
                logger.warning(f"Overwriting {name} in Answer Evaluator registry")
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        ranker_name: AgentRankerTypes,
        config: AgentRankerConfig | None = None,
        **kwargs,
    ) -> AgentRanker:
        if ranker_name not in cls.registry:
            raise ValueError(f"Unknown Agent Ranker {ranker_name}")
        if config is None:
            class_ = cls.registry[ranker_name]
            type_config = class_.get_config_class()
            valid_keys = [field for field in type_config.get_model_fields()]
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            config = type_config(**valid_args)
        return cls.registry[ranker_name].from_config(config)


def get_agent_ranker(
    ranker_name: AgentRankerTypes | str | None = None,
    config: AgentRankerConfig | None = None,
    **kwargs,
) -> AgentRanker:
    if ranker_name is None:
        if config is None:
            raise ValueError("Either ranker_name or config should be provided")
        ranker_name = config.ranker_name
    if isinstance(ranker_name, str):
        ranker_name = AgentRankerTypes(ranker_name)
    return AgentRankerFactory.create(ranker_name, config, **kwargs)
