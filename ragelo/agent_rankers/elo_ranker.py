from __future__ import annotations

import random
from typing import Optional

import numpy as np

from ragelo.agent_rankers.base_agent_ranker import AgentRanker, AgentRankerFactory
from ragelo.evaluators.answer_evaluators import BaseAnswerEvaluator
from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator
from ragelo.logger import logger
from ragelo.types.configurations import EloAgentRankerConfig
from ragelo.types.experiment import Experiment
from ragelo.types.query import Query
from ragelo.types.results import AnswerEvaluatorResult, EloTournamentResult
from ragelo.types.types import AgentRankerTypes


@AgentRankerFactory.register(AgentRankerTypes.ELO)
class EloRanker(AgentRanker):
    name: str = "Elo Agent Ranker"
    config: EloAgentRankerConfig

    def __init__(self, config: EloAgentRankerConfig):
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
        self.games: list[tuple[str, str, str]] = []

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
        experiment.add_evaluation(result, should_print=True)
        return result

    def get_agents_ratings(self):
        return self.agents_scores

    def get_ranked_agents(self) -> list[tuple[str, float]]:
        ranking = sorted(self.get_agents_ratings().items(), key=lambda x: x[1], reverse=True)
        return [(agent, rating) for agent, rating in ranking]

    def run_tournament(self) -> dict[str, int]:
        agents_scores: dict[str, int] = {}
        games: list[tuple[str, str, float]] = []
        for _, agent_a, agent_b, score in self.evaluations:
            score_val = self.score_map[score]
            games.append((agent_a, agent_b, score_val))
        random.shuffle(games)
        for agent_a, agent_b, score_val in games:
            if self.config.verbose:
                logger.info(f"Game: {agent_a} vs {agent_b} -> {score_val}")
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

    def get_agent_losses(self, agent: str) -> list[str, str]:
        """For a given agent, returns a list of tuples(qid, agent) with the query ids and the agents that the agent lost to"""
        lost_games = []
        for qid, agent_a, agent_b, winner in self.games:
            if agent_b == agent and winner == "A":
                lost_games.append((qid, agent_a))
            elif agent_a == agent and winner == "B":
                lost_games.append((qid, agent_b))
        return lost_games

    def run_single_game(
        self,
        query: Query,
        agent_a: str,
        agent_b: str,
        answer_evaluator: BaseAnswerEvaluator,
        retrieval_evaluator: Optional[BaseRetrievalEvaluator] = None,
    ) -> tuple[AnswerEvaluatorResult, AnswerEvaluatorResult]:
        """Run a single game between two agents"""
        assert answer_evaluator.config.pairwise, "Answer evaluator must be pairwise"
        assert agent_a in query.answers, f"Agent {agent_a} not found in query"
        assert agent_b in query.answers, f"Agent {agent_b} not found in query"

        answer_a = query.answers[agent_a]
        answer_b = query.answers[agent_b]

        if retrieval_evaluator:
            retrieval_evaluator.evaluate_all_evaluables(query, n_threads=10)
        evaluation_a_b = answer_evaluator.evaluate(query, answer_a, answer_b)
        evaluation_b_a = answer_evaluator.evaluate(query, answer_b, answer_a)

        winner_a_b = evaluation_a_b.answer.winner
        winner_b_a = evaluation_b_a.answer.winner
        if winner_a_b == winner_b_a:
            winner = winner_a_b
        else:
            winner = "C"
        score_val = self.score_map[winner]
        # update dictionaries and rankings
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
        return evaluation_a_b, evaluation_b_a

    def update_rankings(self, agent_a: str, agent_b: str, score_val: float) -> tuple[int, int]:
        agent_a_rating = self.agents_scores.get(agent_a, self.initial_score)
        agent_b_rating = self.agents_scores.get(agent_b, self.initial_score)

        expected_score = 1 / (1 + 10 ** ((agent_a_rating - agent_b_rating) / 400))
        self.agents_scores[agent_a] = int(agent_a_rating + self.k * (score_val - expected_score))
        self.agents_scores[agent_b] = int(agent_b_rating + self.k * ((1 - score_val) - (1 - expected_score)))
        self.games_played[agent_a] = self.games_played.get(agent_a, 0) + 1
        self.games_played[agent_b] = self.games_played.get(agent_b, 0) + 1
        return self.agents_scores[agent_a], self.agents_scores[agent_b]
