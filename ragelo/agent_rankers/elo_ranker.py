from __future__ import annotations

import random

import numpy as np

from ragelo.agent_rankers.base_agent_ranker import AgentRanker, AgentRankerFactory
from ragelo.logger import logger
from ragelo.types.configurations import EloAgentRankerConfig
from ragelo.types.queries import Queries
from ragelo.types.results import EloTournamentResult
from ragelo.types.types import AgentRankerTypes


@AgentRankerFactory.register(AgentRankerTypes.ELO)
class EloRanker(AgentRanker):
    name: str = "Elo Agent Ranker"
    config: EloAgentRankerConfig

    def __init__(
        self,
        config: EloAgentRankerConfig,
    ):
        super().__init__(config)
        self.score_map = {"A": 1, "B": 0, "C": 0.5}

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

    def run(self, queries: Queries) -> EloTournamentResult:
        """Compute score for each agent"""
        self.evaluations = self._flatten_evaluations(queries)
        agent_scores: dict[str, list[int]] = {}
        for _ in range(self.config.tournaments):
            results = self.run_tournament()
            for agent, score in results.items():
                agent_scores[agent] = agent_scores.get(agent, []) + [score]
        for a in agent_scores:
            self.std_dev[a] = float(np.std(agent_scores[a]))
            self.agents_scores[a] = float(np.mean(agent_scores[a]))

        self.print_ranking()
        result = EloTournamentResult(
            agents=list(self.agents_scores.keys()),
            scores=self.agents_scores,
            games_played=self.games_played,
            wins=self.wins,
            loses=self.losses,
            ties=self.ties,
            total_games=self.total_games,
            total_tournaments=self.config.tournaments,
            std_dev={},
        )
        queries.add_evaluation(result, False)
        return result

    def get_agents_ratings(self):
        if not self.computed:
            raise ValueError("Ranking not computed yet, Run run() first")
        return self.agents_scores

    def run_tournament(self) -> dict[str, int]:
        agents_scores: dict[str, int] = {}
        games: list[tuple[str, str, float]] = []
        for agent_a, agent_b, score in self.evaluations:
            score_val = self.score_map[score]
            games.append((agent_a, agent_b, score_val))
        random.shuffle(games)
        for agent_a, agent_b, score_val in games:
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
            agents_scores[agent_a] = int(
                agent_a_rating + self.k * (score_val - expected_score)
            )
            agents_scores[agent_b] = int(
                agent_b_rating + self.k * ((1 - score_val) - (1 - expected_score))
            )
            self.total_games += 1
            self.games_played[agent_a] = self.games_played.get(agent_a, 0) + 1
            self.games_played[agent_b] = self.games_played.get(agent_b, 0) + 1

        self.computed = True
        return agents_scores

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
                    std_dev = self.std_dev.get(agent, 0)
                    rich.print(
                        f"[bold white]{agent:<15}[/bold white]: {rating:.1f}(±{std_dev:.1f})"
                    )
            except ImportError:
                logger.warning("Rich not installed. Using plain print")
                self.config.rich_print = False
        if not self.config.rich_print:
            print(f"------- Agent Scores by {self.name} -------")
            for agent, rating in scores:
                std_dev = self.std_dev.get(agent, 0)
                print(f"{agent:<15}: {rating:.1f} (±{std_dev:.1f})")
