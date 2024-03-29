import random

from ragelo.agent_rankers.base_agent_ranker import AgentRanker, AgentRankerFactory
from ragelo.logger import logger
from ragelo.types import EloAgentRankerConfig


@AgentRankerFactory.register("elo")
class EloRanker(AgentRanker):
    name: str = "Elo Agent Ranker"
    config: EloAgentRankerConfig

    def __init__(
        self,
        config: EloAgentRankerConfig,
        evaluations: list[tuple[str, str, str]],
    ):
        super().__init__(config, evaluations)
        self.score_map = {"A": 1, "B": 0, "C": 0.5}

        self.agents: dict[str, int] = {}
        self.games: list[tuple[str, str, float]] = []
        self.computed = False
        self.initial_score = self.config.initial_score
        self.k = self.config.k

    def run(self):
        """Compute score for each agent"""
        self.games, self.agents = self.__get_elo_scores()

        while self.games:
            self.__play_one_game()
        self.dump_ranking()
        self.print_ranking()

    def get_agents_ratings(self):
        if not self.computed:
            raise ValueError("Ranking not computed yet, Run evaluate() first")
        return self.agents

    def __get_elo_scores(self) -> tuple[list[tuple[str, str, float]], dict[str, int]]:
        games: list[tuple[str, str, float]] = []
        agents = {}
        for agent_a, agent_b, score in self.evaluations:
            score_val = self.score_map[score]
            games.append((agent_a, agent_b, score_val))
            logger.info(f"Game: {agent_a} vs {agent_b} -> {score_val}")
            if agent_a not in agents:
                agents[agent_a] = self.initial_score
            if agent_b not in agents:
                agents[agent_b] = self.initial_score
        random.shuffle(games)
        self.computed = True
        return games, agents

    def __play_one_game(self):
        player1, player2, result = self.games.pop()
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
