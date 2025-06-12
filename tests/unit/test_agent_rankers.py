from __future__ import annotations


from ragelo.agent_rankers.base_agent_ranker import get_agent_ranker
from ragelo.agent_rankers.elo_ranker import EloRanker
from ragelo.types.configurations import EloAgentRankerConfig
from ragelo.types.results import EloTournamentResult


class TestEloRanker:
    def test_get_agent_ranker(self):
        """get_agent_ranker should return an EloRanker instance when called with 'elo'."""
        ranker = get_agent_ranker("elo")
        assert isinstance(ranker, EloRanker)

    def test_elo_ranker_run(self, experiment, pairwise_answer_evaluation):
        """Running the EloRanker on an experiment with one evaluated game should
        1. return an EloTournamentResult
        2. correctly update wins / losses / games_played counters
        3. assign a higher rating to the winner
        """
        # Add a single pair-wise evaluation (Agent A wins) to the experiment
        experiment.add_evaluation(pairwise_answer_evaluation, should_save=False)

        # Create ranker with a single tournament to make the expected values deterministic
        config = EloAgentRankerConfig(tournaments=1, verbose=False)
        ranker = EloRanker.from_config(config)

        result = ranker.run(experiment)

        # Basic type checks
        assert isinstance(result, EloTournamentResult)
        assert set(result.agents) == {"agent1", "agent2"}

        # The winner (agent1) should have exactly one win and no losses
        assert result.wins["agent1"] == 1
        assert result.loses.get("agent1", 0) == 0

        # The loser (agent2) should have exactly one loss and no wins
        assert result.loses["agent2"] == 1
        assert result.wins.get("agent2", 0) == 0

        # Each agent played exactly one game
        assert result.games_played["agent1"] == 1
        assert result.games_played["agent2"] == 1
        assert result.total_games == 1
        assert result.total_tournaments == 1

        # The winner must have a higher Elo score than the loser
        assert result.scores["agent1"] > result.scores["agent2"]
