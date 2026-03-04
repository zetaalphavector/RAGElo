from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragelo.agent_rankers.elo_ranker import EloRanker
from ragelo.types.answer_formats import PairwiseEvaluationAnswer
from ragelo.types.configurations import EloAgentRankerConfig
from ragelo.types.results import PairwiseGameEvaluatorResult


class TestEloRanker:
    @pytest.fixture
    def elo_ranker(self):
        config = EloAgentRankerConfig()
        return EloRanker(config)

    def test_add_new_agent(self, elo_ranker):
        result = elo_ranker.add_new_agent("agent_x")
        assert result == elo_ranker.initial_score
        assert elo_ranker.agents_scores["agent_x"] == elo_ranker.initial_score
        assert elo_ranker.games_played["agent_x"] == 0
        assert elo_ranker.wins["agent_x"] == 0
        assert elo_ranker.losses["agent_x"] == 0
        assert elo_ranker.ties["agent_x"] == 0

    def test_update_rankings(self, elo_ranker):
        elo_ranker.add_new_agent("agent_a")
        elo_ranker.add_new_agent("agent_b")
        new_a, new_b = elo_ranker.update_rankings("agent_a", "agent_b", 1.0)
        assert new_a > elo_ranker.initial_score
        assert new_b < elo_ranker.initial_score
        assert elo_ranker.games_played["agent_a"] == 1
        assert elo_ranker.games_played["agent_b"] == 1

    def test_update_rankings_tie(self, elo_ranker):
        elo_ranker.add_new_agent("agent_a")
        elo_ranker.add_new_agent("agent_b")
        new_a, new_b = elo_ranker.update_rankings("agent_a", "agent_b", 0.5)
        assert new_a == elo_ranker.initial_score
        assert new_b == elo_ranker.initial_score

    def test_get_agent_losses(self, elo_ranker):
        elo_ranker.games = [
            ("q1", "agent_a", "agent_b", "A"),
            ("q2", "agent_a", "agent_b", "B"),
            ("q3", "agent_a", "agent_c", "A"),
            ("q4", "agent_b", "agent_c", "C"),
        ]
        losses_a = elo_ranker.get_agent_losses("agent_a")
        assert losses_a == [("q2", "agent_b")]

        losses_b = elo_ranker.get_agent_losses("agent_b")
        assert losses_b == [("q1", "agent_a")]

        losses_c = elo_ranker.get_agent_losses("agent_c")
        assert losses_c == [("q3", "agent_a")]

    def test_get_agent_losses_no_losses(self, elo_ranker):
        elo_ranker.games = [
            ("q1", "agent_a", "agent_b", "A"),
        ]
        assert elo_ranker.get_agent_losses("agent_a") == []

    @pytest.mark.asyncio
    async def test_run_single_game(self, elo_ranker, experiment):
        query = experiment.queries["0"]
        mock_evaluator = MagicMock()
        mock_evaluator.config.pairwise = True

        mock_result = PairwiseGameEvaluatorResult(
            qid="0",
            agent_a="agent1",
            agent_b="agent2",
            evaluator_name="pairwise",
            answer=PairwiseEvaluationAnswer(
                answer_a_analysis="A is good",
                answer_b_analysis="B is bad",
                comparison_reasoning="A is better",
                winner="A",
            ),
        )
        mock_evaluator.evaluate_async = AsyncMock(return_value=mock_result)

        result = await elo_ranker.run_single_game(
            query=query,
            agent_a="agent1",
            agent_b="agent2",
            answer_evaluator=mock_evaluator,
        )

        assert result.winner == "A"
        assert len(elo_ranker.games) == 1
        assert elo_ranker.games[0] == ("0", "agent1", "agent2", "A")
        assert elo_ranker.wins["agent1"] == 1
        assert elo_ranker.losses["agent2"] == 1
        assert elo_ranker.games_played["agent1"] == 1
        assert elo_ranker.games_played["agent2"] == 1

    @pytest.mark.asyncio
    async def test_run_single_game_updates_rankings(self, elo_ranker, experiment):
        query = experiment.queries["0"]
        mock_evaluator = MagicMock()
        mock_evaluator.config.pairwise = True

        mock_result = PairwiseGameEvaluatorResult(
            qid="0",
            agent_a="agent1",
            agent_b="agent2",
            evaluator_name="pairwise",
            answer=PairwiseEvaluationAnswer(
                answer_a_analysis="A is good",
                answer_b_analysis="B is bad",
                comparison_reasoning="A is better",
                winner="A",
            ),
        )
        mock_evaluator.evaluate_async = AsyncMock(return_value=mock_result)

        await elo_ranker.run_single_game(
            query=query,
            agent_a="agent1",
            agent_b="agent2",
            answer_evaluator=mock_evaluator,
        )

        assert elo_ranker.agents_scores["agent1"] > elo_ranker.initial_score
        assert elo_ranker.agents_scores["agent2"] < elo_ranker.initial_score
