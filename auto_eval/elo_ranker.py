import csv
import random

from rich import print


class TournamentEloRating:
    def __init__(
        self,
        games: str,
        k: int = 32,
        initial_score: int = 1000,
    ):
        self.k = k
        self.initial_score = initial_score
        self.__games, self.__players = self.__read_games(games)

    def __read_games(self, file_path: str):
        games = []
        players = {}
        for row in csv.DictReader(open(file_path)):
            agent_a = row["agent_a"]
            agent_b = row["agent_b"]
            score = float(row["score"])
            games.append((agent_a, agent_b, score))
            if agent_a not in players:
                players[agent_a] = self.initial_score
            if agent_b not in players:
                players[agent_b] = self.initial_score
        print(f"Loaded {len(games)} games")
        random.shuffle(games)
        return games, players

    def __expected_score(self, rating1, rating2):
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

    def __update_rating(self, rating, expected_score, actual_score):
        return rating + self.k * (actual_score - expected_score)

    def get_player_ratings(self):
        return self.__players

    def __play_one_game(self):
        player1, player2, result = self.__games.pop()
        player1_rating = self.__players[player1]
        player2_rating = self.__players[player2]
        expected_score = self.__expected_score(player1_rating, player2_rating)

        self.__players[player1] = self.__update_rating(
            player1_rating, expected_score, result
        )
        self.__players[player2] = self.__update_rating(
            player2_rating, 1 - expected_score, 1 - result
        )

    def play_all_games(self):
        while self.__games:
            self.__play_one_game()
