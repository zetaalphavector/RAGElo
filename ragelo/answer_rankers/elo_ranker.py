import random
from typing import Dict, List, Tuple

from ragelo.answer_rankers.base_answer_ranker import AnswerRanker, AnswerRankerFactory
from ragelo.logger import logger


@AnswerRankerFactory.register("elo")
class EloRanker(AnswerRanker):
    def __init__(self, initial_score: int = 1000, k: int = 32, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_map = {"A": 1, "B": 0, "C": 0.5}
        self.name = "Elo ranking"
        self.players: Dict[str, float] = {}
        self.games: List[Tuple[str, str, float]] = []
        self.computed = False
        self.initial_score = initial_score
        self.k = k

    def evaluate(self):
        """Compute score for each agent"""
        self.games, self.players = self.__get_elo_scores()
        while self.games:
            self.__play_one_game()

    def get_agents_ratings(self):
        if not self.computed:
            raise ValueError("Ranking not computed yet, Run evaluate() first")
        return self.players

    def __get_elo_scores(self) -> Tuple[List[Tuple[str, str, float]], Dict[str, int]]:
        games: List[Tuple[str, str, float]] = []
        players = {}
        for agent_a, agent_b, score in self.evaluations:
            score_val = self.score_map[score]
            games.append((agent_a, agent_b, score_val))
            logger.info(f"Game: {agent_a} vs {agent_b} -> {score_val}")
            if agent_a not in players:
                players[agent_a] = self.initial_score
            if agent_b not in players:
                players[agent_b] = self.initial_score
        random.shuffle(games)
        self.computed = True
        return games, players

    def __play_one_game(self):
        player1, player2, result = self.games.pop()
        player1_rating = self.players[player1]
        player2_rating = self.players[player2]
        expected_score = self.__expected_score(player1_rating, player2_rating)

        self.players[player1] = self.__update_rating(
            player1_rating, expected_score, result
        )
        self.players[player2] = self.__update_rating(
            player2_rating, 1 - expected_score, 1 - result
        )

    def __expected_score(self, rating1, rating2):
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

    def __update_rating(self, rating, expected_score, actual_score):
        return rating + self.k * (actual_score - expected_score)
