import random
from typing import Dict, List, Optional, Tuple

from ragelo.agent_rankers.base_agent_ranker import AgentRanker, AgentRankerFactory
from ragelo.logger import logger
from ragelo.types import EloAgentRankerConfig
from ragelo.types.types import AgentRankerTypes, Query


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

        self.agents: Dict[str, int] = {}
        self.games: List[Tuple[str, str, float]] = []
        self.computed = False
        self.initial_score = self.config.initial_score
        self.k = self.config.elo_k

    def run(
        self,
        queries: Optional[List[Query]] = None,
        evaluations_file: Optional[str] = None,
    ) -> Dict[str, int]:
        """Compute score for each agent"""
        queries = self._prepare_queries(queries, evaluations_file)
        self.evaluations = self._flatten_evaluations(queries)
        agent_scores: Dict[str, List[int]] = {}
        for _ in range(self.config.rounds):
            self.games = self.__get_elo_scores()
            while self.games:
                player1, player2, result = self.games.pop()
                self.__play_one_game(player1, player2, result)
            for a in self.agents:
                if a not in agent_scores:
                    agent_scores[a] = []
                agent_scores[a].append(self.agents[a])
                self.agents[a] = self.initial_score
        # Average the scores over all rounds
        for a in agent_scores:
            self.agents[a] = sum(agent_scores[a]) // len(agent_scores[a])
        self.dump_ranking()
        self.print_ranking()
        return self.agents

    def get_agents_ratings(self):
        if not self.computed:
            raise ValueError("Ranking not computed yet, Run run() first")
        return self.agents

    def __get_elo_scores(self) -> List[Tuple[str, str, float]]:
        games: List[Tuple[str, str, float]] = []
        for agent_a, agent_b, score in self.evaluations:
            score_val = self.score_map[score]
            games.append((agent_a, agent_b, score_val))
            logger.info(f"Game: {agent_a} vs {agent_b} -> {score_val}")
            if agent_a not in self.agents:
                self.agents[agent_a] = self.initial_score
            if agent_b not in self.agents:
                self.agents[agent_b] = self.initial_score
        random.shuffle(games)
        self.computed = True
        return games

    def __play_one_game(self, player1, player2, result):
        player1_rating = self.agents[player1]
        player2_rating = self.agents[player2]
        expected_score = self.__expected_score(player1_rating, player2_rating)

        self.agents[player1] = self.__update_rating(
            player1_rating, expected_score, result
        )
        self.agents[player2] = self.__update_rating(
            player2_rating, 1 - expected_score, 1 - result
        )

    def __expected_score(self, rating1, rating2):
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

    def __update_rating(self, rating, expected_score, actual_score) -> int:
        return int(rating + self.k * (actual_score - expected_score))
