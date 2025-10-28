from __future__ import annotations

import logging
import random

import numpy as np

from ragelo.agent_rankers.base_agent_ranker import AgentRanker, AgentRankerFactory
from ragelo.evaluators.answer_evaluators.pairwise_evaluator import PairwiseAnswerEvaluator
from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import BaseRetrievalEvaluator
from ragelo.types import EloTournamentResult, Experiment, Query
from ragelo.types.configurations import EloAgentRankerConfig
from ragelo.types.results import PairwiseGameEvaluatorResult
from ragelo.types.types import AgentRankerTypes

logger = logging.getLogger(__name__)


@AgentRankerFactory.register(AgentRankerTypes.ELO)
class EloRanker(AgentRanker):
    name: str = "Elo Agent Ranker"
    config: EloAgentRankerConfig

    def __init__(
        self,
        config: EloAgentRankerConfig,
    ):
        super().__init__(config)
        self.score_map = config.score_mapping

        self.agents_scores: dict[str, float] = {}
        self.wins: dict[str, int] = {}
        self.losses: dict[str, int] = {}
        self.ties: dict[str, int] = {}
        self.total_games: int = 0
        self.games_played: dict[str, int] = {}
        self.computed: bool = False
        self.initial_score: int = self.config.initial_score
        self.k: int = self.config.elo_k
        self.std_dev: dict[str, float] = {}
        self.games: list[tuple[str, str, str, str]] = []

    def add_new_agent(self, agent: str):
        if agent not in self.agents_scores:
            self.agents_scores[agent] = self.initial_score
            self.games_played[agent] = 0
            self.wins[agent] = 0
            self.losses[agent] = 0
            self.ties[agent] = 0
            self.std_dev[agent] = 0

    def run(self, experiment: Experiment) -> EloTournamentResult:
        """Compute score for each agent"""
        self.evaluations = self._flatten_evaluations(experiment)
        agent_scores: dict[str, list[int]] = {}
        for _ in range(self.config.tournaments):
            results = self.run_tournament()
            for agent, score in results.items():
                agent_scores[agent] = agent_scores.get(agent, []) + [score]
        for a in agent_scores:
            self.std_dev[a] = float(np.std(agent_scores[a]))
            self.agents_scores[a] = float(np.mean(agent_scores[a]))

        result = EloTournamentResult(
            agents=list(self.agents_scores.keys()),
            scores=self.agents_scores,
            games_played=self.games_played,
            wins=self.wins,
            loses=self.losses,
            ties=self.ties,
            std_dev=self.std_dev,
            total_games=self.total_games,
            total_tournaments=self.config.tournaments,
        )
        experiment.add_evaluation(None, result, should_print=self.config.render)
        return result

    def get_agents_ratings(self):
        return self.agents_scores

    def get_ranked_agents(self) -> list[tuple[str, float]]:
        ranking = sorted(self.get_agents_ratings().items(), key=lambda x: x[1], reverse=True)
        return [(agent, rating) for agent, rating in ranking]

    def run_tournament(self) -> dict[str, int]:
        agents_scores: dict[str, int] = {}
        games: list[tuple[str, str, float]] = []
        for agent_a, agent_b, score in self.evaluations:
            score_val = self.score_map[score]
            games.append((agent_a, agent_b, score_val))
        random.shuffle(games)
        for agent_a, agent_b, score_val in games:
            logger.debug(f"Game: {agent_a} vs {agent_b} -> {score_val}")
            if score_val == 1:
                self.wins[agent_a] = self.wins.get(agent_a, 0) + 1
                self.losses[agent_b] = self.losses.get(agent_b, 0) + 1
            elif score_val == 0:
                self.wins[agent_b] = self.wins.get(agent_b, 0) + 1
                self.losses[agent_a] = self.losses.get(agent_a, 0) + 1
            else:
                self.ties[agent_a] = self.ties.get(agent_a, 0) + 1
                self.ties[agent_b] = self.ties.get(agent_b, 0) + 1
            agent_a_rating = agents_scores.get(agent_a, self.initial_score)
            agent_b_rating = agents_scores.get(agent_b, self.initial_score)

            expected_score = 1 / (1 + 10 ** ((agent_a_rating - agent_b_rating) / 400))
            agents_scores[agent_a] = int(agent_a_rating + self.k * (score_val - expected_score))
            agents_scores[agent_b] = int(agent_b_rating + self.k * ((1 - score_val) - (1 - expected_score)))
            self.total_games += 1
            self.games_played[agent_a] = self.games_played.get(agent_a, 0) + 1
            self.games_played[agent_b] = self.games_played.get(agent_b, 0) + 1

        self.computed = True
        return agents_scores

    async def run_single_game(
        self,
        query: Query,
        agent_a: str,
        agent_b: str,
        evaluator: PairwiseAnswerEvaluator,
        retrieval_evaluator: BaseRetrievalEvaluator | None = None,
        experiment: Experiment | None = None,
    ) -> tuple[int, int]:
        assert agent_a in query.answers, f"Agent {agent_a} not found in query"
        assert agent_b in query.answers, f"Agent {agent_b} not found in query"
        game = query.add_pairwise_game(agent_a, agent_b, exist_ok=True)

        if retrieval_evaluator:
            retrieval_evaluator.evaluate_all_evaluables(query, n_threads=10)
            if experiment:
                experiment.save()
        evaluation: PairwiseGameEvaluatorResult = await evaluator.evaluate_async((query, game))
        if experiment:
            experiment.add_evaluation((query, game), evaluation, exist_ok=True)
        winner = evaluation.winner
        score_val = self.score_map[winner]
        agent_a = game.agent_a_answer.agent
        agent_b = game.agent_b_answer.agent
        if winner == "A":
            self.wins[agent_a] = self.wins.get(agent_a, 0) + 1
            self.losses[agent_b] = self.losses.get(agent_b, 0) + 1
        elif winner == "B":
            self.wins[agent_b] = self.wins.get(agent_b, 0) + 1
            self.losses[agent_a] = self.losses.get(agent_a, 0) + 1
        else:
            self.ties[agent_a] = self.ties.get(agent_a, 0) + 1
            self.ties[agent_b] = self.ties.get(agent_b, 0) + 1
        self.games.append((query.qid, agent_a, agent_b, winner))
        self.update_rankings(agent_a, agent_b, score_val)
        return self.agents_scores[agent_a], self.agents_scores[agent_b]

    def update_rankings(self, agent_a: str, agent_b: str, score_val: float) -> tuple[int, int]:
        agent_a_rating = self.agents_scores.get(agent_a, self.initial_score)
        agent_b_rating = self.agents_scores.get(agent_b, self.initial_score)

        expected_score = 1 / (1 + 10 ** ((agent_a_rating - agent_b_rating) / 400))
        self.agents_scores[agent_a] = int(agent_a_rating + self.k * (score_val - expected_score))
        self.agents_scores[agent_b] = int(agent_b_rating + self.k * ((1 - score_val) - (1 - expected_score)))
        self.games_played[agent_a] = self.games_played.get(agent_a, 0) + 1
        self.games_played[agent_b] = self.games_played.get(agent_b, 0) + 1
        return self.agents_scores[agent_a], self.agents_scores[agent_b]
