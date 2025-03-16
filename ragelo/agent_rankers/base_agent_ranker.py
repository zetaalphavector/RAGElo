from __future__ import annotations

from collections import defaultdict
from typing import Any, Type, get_type_hints

from ragelo.logger import logger
from ragelo.types.configurations.agent_ranker_configs import AgentRankerConfig
from ragelo.types.evaluables import FlatGame
from ragelo.types.experiment import Experiment
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

    def run(self, experiment: Experiment) -> Any:
        """Computes a score for each agent in the experiment"""
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

    def get_documents_retrieved_by_agent(self, experiment: Experiment, qid: str, agent: str) -> list[str]:
        """Returns the list of documents used by the agent to answer the query"""
        retrieved_docs = []
        for doc_id, doc in experiment[qid].retrieved_docs.items():
            if agent in doc.retrieved_by:
                retrieved_docs.append(doc_id)
        return retrieved_docs

    def get_cited_documents(self, experiment, qid: str, agent: str) -> list[str]:
        documents_retrieved = self.get_documents_retrieved_by_agent(experiment, qid, agent)
        if len(documents_retrieved) == 0:
            return []
        cited_docs = []
        agent_answer = experiment[qid].answers[agent].text
        for did in documents_retrieved:
            if did in agent_answer:
                cited_docs.append(did)
        return cited_docs

    def get_tournament_statistics(self, evaluations: list[FlatGame]):
        games_per_agent: dict[str, int] = defaultdict(lambda: 0)
        games_per_query: dict[str, int] = defaultdict(lambda: 0)
        for g in evaluations:
            games_per_query[g.qid] += 1
            games_per_agent[g.agent_a] += 1
            games_per_agent[g.agent_b] += 1

        logger.info(f"Total games: {len(evaluations)}")
        for agent, games in games_per_agent.items():
            logger.info(f"Agent {agent} played {games} games")
        logger.info(f"{len(games_per_query)} queries had at least one game")
        for query, games in games_per_query.items():
            logger.info(f"Query {query} had {games} games")

    def _flatten_evaluations(
        self,
        experiment: Experiment,
        ignore_bad_answers: bool = True,
        only_queries_with_all_agents: bool = False,
    ) -> list[FlatGame]:
        evaluations = []
        warned_agents = set()
        for query in experiment:
            for game in query.pairwise_games:
                if game.evaluation is not None:
                    if ignore_bad_answers:
                        cited_docs = self.get_cited_documents(experiment, query.qid, game.agent_a_answer.agent)
                        if len(cited_docs) == 0:
                            if (
                                query.qid,
                                game.agent_a_answer.agent,
                            ) not in warned_agents:
                                logger.warning(
                                    f"Agent {game.agent_a_answer.agent} did not cite any document for query"
                                    f" {query.qid} Skipping evaluation for this agent"
                                )
                                warned_agents.add((query.qid, game.agent_a_answer.agent))
                            continue

                    evaluations.append(
                        FlatGame(
                            qid=query.qid,
                            agent_a=game.agent_a_answer.agent,
                            agent_b=game.agent_b_answer.agent,
                            evaluation=str(game.evaluation.answer),
                        )
                    )

        games_per_agent: dict[str, int] = defaultdict(lambda: 0)
        games_per_query: dict[str, int] = defaultdict(lambda: 0)
        for g in evaluations:
            games_per_query[g.qid] += 1
            games_per_agent[g.agent_a] += 1
            games_per_agent[g.agent_b] += 1
        queries_with_all_agents = {qid for qid, games in games_per_query.items() if games % len(games_per_agent) == 0}

        if only_queries_with_all_agents:
            evaluations = [e for e in evaluations if e.qid in queries_with_all_agents]
        self.get_tournament_statistics(evaluations)
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
