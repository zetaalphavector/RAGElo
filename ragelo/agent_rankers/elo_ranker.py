from __future__ import annotations

import random
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

from ragelo.agent_rankers.base_agent_ranker import AgentRanker, AgentRankerFactory
from ragelo.types.configurations import EloAgentRankerConfig
from ragelo.types.evaluables import FlatGame
from ragelo.types.experiment import Experiment
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
        self.score_map = config.score_mapping

        self.agents_scores: dict[str, float] = {}
        self.agents_rankings: dict[str, int] = {}
        self.wins: dict[str, int] = {}
        self.losses: dict[str, int] = {}
        self.ties: dict[str, int] = {}
        self.total_games: int = 0
        self.games_played: dict[str, int] = {}
        self.computed: bool = False
        self.initial_score: int = self.config.initial_score
        self.k: int = self.config.elo_k
        self.std_dev: dict[str, float] = {}
        self.pairwise_wins: dict[str, dict[str, int]] = {}  # new attribute for pairwise wins

    def run(self, experiment: Experiment) -> EloTournamentResult:
        """Compute score for each agent"""
        evaluations = self._flatten_evaluations(experiment)
        all_agents = {game.agent_a for game in evaluations} | {game.agent_b for game in evaluations}
        agent_scores: dict[str, list[float]] = {a: [] for a in all_agents}

        for _ in range(self.config.tournaments):
            if self.config.tournament_per_query:
                results = self.run_tournament_per_query(evaluations, all_agents)
            else:
                results = self.run_tournament(evaluations, all_agents)
            for agent, score in results.items():
                agent_scores[agent].append(score)
        for a, scores in agent_scores.items():
            self.std_dev[a] = int(sem(scores))
            self.agents_scores[a] = int(np.mean(scores))

        result = EloTournamentResult(
            agents=list(all_agents),
            scores=self.agents_scores,
            games_played=self.games_played,
            wins=self.wins,
            loses=self.losses,
            ties=self.ties,
            std_dev=self.std_dev,
            total_games=self.total_games,
            total_tournaments=self.config.tournaments,
        )
        experiment.add_evaluation(result)
        return result

    def get_agents_ratings(self):
        return self.agents_scores

    def get_ranked_agents(self) -> list[tuple[str, float]]:
        ranking = sorted(self.get_agents_ratings().items(), key=lambda x: x[1], reverse=True)
        return [(agent, rating) for agent, rating in ranking]

    def run_tournament(self, evaluations: list[FlatGame], agents: Iterable[str]) -> dict[str, float]:
        agents_scores: dict[str, float] = {a: 0.0 for a in agents}
        games: list[tuple[str, str, float]] = []
        for game in evaluations:
            score_val = self.score_map[game.evaluation]
            games.append((game.agent_a, game.agent_b, score_val))
        random.shuffle(games)
        for agent_a, agent_b, score_val in games:
            # logger.info(f"Game: {agent_a} vs {agent_b} -> {score_val}")
            if score_val == 1:
                self.wins[agent_a] = self.wins.get(agent_a, 0) + 1
                self.losses[agent_b] = self.losses.get(agent_b, 0) + 1
                self.pairwise_wins.setdefault(agent_a, {})[agent_b] = (
                    self.pairwise_wins.get(agent_a, {}).get(agent_b, 0) + 1
                )

            elif score_val == 0:
                self.wins[agent_b] = self.wins.get(agent_b, 0) + 1
                self.losses[agent_a] = self.losses.get(agent_a, 0) + 1
                self.pairwise_wins.setdefault(agent_b, {})[agent_a] = (
                    self.pairwise_wins.get(agent_b, {}).get(agent_a, 0) + 1
                )

            else:
                self.ties[agent_a] = self.ties.get(agent_a, 0) + 1
                self.ties[agent_b] = self.ties.get(agent_b, 0) + 1
            agent_a_rating = agents_scores.get(agent_a, self.initial_score)
            agent_b_rating = agents_scores.get(agent_b, self.initial_score)

            expected_score = 1 / (1 + 10 ** ((agent_a_rating - agent_b_rating) / 400))
            agents_scores[agent_a] = max(int(agent_a_rating + self.k * (score_val - expected_score)), 0)
            agents_scores[agent_b] = max(
                int(agent_b_rating + self.k * ((1 - score_val) - (1 - expected_score))),
                0,
            )
            self.total_games += 1
            self.games_played[agent_a] = self.games_played.get(agent_a, 0) + 1
            self.games_played[agent_b] = self.games_played.get(agent_b, 0) + 1

        self.computed = True
        return agents_scores

    def run_tournament_per_query(self, evaluations: list[FlatGame], agents: Iterable[str]) -> dict[str, float]:
        agents_scores: dict[str, float] = {a: self.initial_score for a in agents}
        games_per_query: dict[str, list[FlatGame]] = {}
        for game in evaluations:
            games_per_query[game.qid] = games_per_query.get(game.qid, []) + [game]
        qids = list(games_per_query.keys())
        random.shuffle(qids)
        for qid in qids:
            actual_scores: dict[str, list[float]] = {a: [] for a in agents}
            expected_scores: dict[str, list[float]] = {a: [] for a in agents}
            for game in games_per_query[qid]:
                score_val = self.score_map[game.evaluation]
                agent_a = game.agent_a
                agent_b = game.agent_b
                if score_val == 1:
                    self.wins[agent_a] = self.wins.get(agent_a, 0) + 1
                    self.losses[agent_b] = self.losses.get(agent_b, 0) + 1
                elif score_val == 0:
                    self.wins[agent_b] = self.wins.get(agent_b, 0) + 1
                    self.losses[agent_a] = self.losses.get(agent_a, 0) + 1
                else:
                    self.ties[agent_a] = self.ties.get(agent_a, 0) + 1
                    self.ties[agent_b] = self.ties.get(agent_b, 0) + 1
                self.total_games += 1
                self.games_played[agent_a] = self.games_played.get(agent_a, 0) + 1
                self.games_played[agent_b] = self.games_played.get(agent_b, 0) + 1

                r_a = agents_scores[agent_a]
                r_b = agents_scores[agent_b]
                s_a = score_val
                s_b = 1 - score_val
                e_a = 1 / (1 + (10 ** ((r_b - r_a) / 400)))
                e_b = 1 - e_a
                actual_scores[agent_a].append(s_a)
                actual_scores[agent_b].append(s_b)
                expected_scores[agent_a].append(e_a)
                expected_scores[agent_b].append(e_b)
            # update scores
            for agent in agents:
                actual_score = sum(actual_scores[agent])
                expected_score = sum(expected_scores[agent])
                agents_scores[agent] += self.k * (actual_score - expected_score)
        self.computed = True
        return agents_scores

    def plot_pairwise_win_heatmap(self) -> None:
        """
        Generate and display a heatmap that shows the percentage of wins between agents.
        For each pair of agents, the cell shows the percentage of wins for the row agent
        out of the total games played between the two.
        Uses a mapping to convert internal agent names to display names.
        """
        # Mapping from internal agent names to desired names.
        try:
            import pandas as pd
            import seaborn as sns
        except ImportError:
            raise ImportError("seaborn and pandas are required to plot the pairwise win heatmap")
        name_mapping = {
            "hybrid-patents-parallel": "Hybrid",
            "keyword-only-parallel": "GraphQL",
            "knn-only-parallel": "KNN",
        }
        # Get the set of all agents involved in pairwise matches.
        agents_set: set[str] = set()
        for winner, losers in self.pairwise_wins.items():
            agents_set.add(winner)
            agents_set |= set(losers.keys())
        agents = sorted(list(agents_set))
        # Create a matrix of wins initialized to zero.
        matrix = pd.DataFrame(0, index=agents, columns=agents)
        for winner, losers in self.pairwise_wins.items():
            for loser, count in losers.items():
                matrix.at[winner, loser] = count

        # Compute percentage matrix.
        percentage_matrix = matrix.copy().astype(float)
        for i in matrix.index:
            for j in matrix.columns:
                if i == j:
                    percentage_matrix.at[i, j] = np.nan  # no self-match percentage
                else:
                    total_games = matrix.at[i, j] + matrix.at[j, i]
                    if total_games > 0:
                        percentage_matrix.at[i, j] = matrix.at[i, j] / total_games * 100
                    else:
                        percentage_matrix.at[i, j] = 0.0

        # Rename index and columns using the mapping.
        percentage_matrix.index = [name_mapping.get(agent, agent) for agent in percentage_matrix.index]  # type: ignore
        percentage_matrix.columns = [name_mapping.get(agent, agent) for agent in percentage_matrix.columns]  # type: ignore

        # Prepare annotations with a percentage format.
        annot = percentage_matrix.applymap(lambda x: f"{x:.1f}%" if pd.notnull(x) else "")  # type: ignore

        # Plot the heatmap.
        sns.heatmap(
            percentage_matrix,
            annot=annot,
            fmt="",
            cmap="YlGnBu",
            mask=percentage_matrix.isnull(),
        )
        plt.title("Pairwise Wins Percentage Heatmap")
        plt.xlabel("Opponent (Loser)")
        plt.ylabel("Winner")
        plt.savefig("pairwise_wins_heatmap.png")
        plt.show()
        plt.show()
