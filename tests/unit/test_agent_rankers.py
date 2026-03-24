from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ragelo.agent_rankers.elo_ranker import EloRanker
from ragelo.types.answer_formats import PairwiseEvaluationAnswer
from ragelo.types.configurations import EloAgentRankerConfig
from ragelo.types.evaluables import AgentAnswer
from ragelo.types.experiment import Experiment
from ragelo.types.results import PairwiseGameEvaluatorResult
from ragelo.types.storage import FileStorageBackend


class TestEloRanker:
    @pytest.fixture
    def elo_ranker(self):
        config = EloAgentRankerConfig()
        return EloRanker(config)

    @pytest.fixture
    def empty_experiment(self, tmp_path):
        return Experiment(
            experiment_name="empty_test",
            storage_backend=FileStorageBackend(tmp_path / "empty.json", tmp_path / "empty_test_results.jsonl"),
        )

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

    def test_expected_score(self, elo_ranker):
        assert elo_ranker._expected_score(1000, 1000) == pytest.approx(0.5)
        assert elo_ranker._expected_score(1200, 1000) > 0.5
        assert elo_ranker._expected_score(1000, 1200) < 0.5
        assert elo_ranker._expected_score(1400, 1000) == pytest.approx(1.0 / (1.0 + 10 ** ((1000 - 1400) / 400.0)))

    def test_opponent_info_value(self, elo_ranker):
        elo_ranker.add_new_agent("opp1")
        elo_ranker.add_new_agent("opp2")
        elo_ranker.games_played["opp1"] = 0
        elo_ranker.games_played["opp2"] = 100
        val1 = elo_ranker._opponent_info_value(1000, "opp1")
        val2 = elo_ranker._opponent_info_value(1000, "opp2")
        # Equal-rated opponent with fewer games should be more informative
        assert val1 > val2

    def test_question_entropy_scores(self, elo_ranker):
        elo_ranker.evaluations = [
            ("q1", "a1", "a2", "A"),
            ("q1", "a1", "a2", "B"),
            ("q2", "a1", "a2", "A"),
            ("q2", "a1", "a2", "A"),
        ]
        scores = elo_ranker._question_entropy_scores()
        # q1 has more entropy (50/50 split) than q2 (100% A)
        assert scores["q1"] > scores["q2"]

    def test_question_entropy_scores_tie_penalty(self, elo_ranker):
        elo_ranker.evaluations = [
            ("q1", "a1", "a2", "C"),
            ("q1", "a1", "a2", "C"),
        ]
        scores = elo_ranker._question_entropy_scores()
        # All ties → tie_penalty = 0 → score = 0
        assert scores["q1"] == 0.0

    def test_wilson_ci(self, elo_ranker):
        low, high = elo_ranker._wilson_ci(10, 20, 1.96)
        assert 0.0 < low < 0.5
        assert 0.5 < high < 1.0
        # CI should contain the true proportion 0.5
        assert low < 0.5 < high

    def test_wilson_ci_zero_games(self, elo_ranker):
        low, high = elo_ranker._wilson_ci(0, 0, 1.96)
        assert low == 0.0
        assert high == 1.0

    @pytest.mark.asyncio
    async def test_add_agent_without_games_no_opponents(self, elo_ranker, empty_experiment):
        mock_evaluator = MagicMock()
        mock_evaluator.config.pairwise = True

        result = await elo_ranker.add_agent_without_games(
            experiment=empty_experiment,
            new_agent="new_agent",
            answer_evaluator=mock_evaluator,
        )

        assert "new_agent" in result.scores
        assert result.scores["new_agent"] == float(elo_ranker.initial_score)
        assert result.games_played["new_agent"] == 0

    @pytest.mark.asyncio
    async def test_add_agent_without_games_plays_games(self, elo_ranker, experiment):
        mock_evaluator = MagicMock()
        mock_evaluator.config.pairwise = True

        call_count = 0

        async def mock_evaluate_async(eval_tuple):
            nonlocal call_count
            call_count += 1
            return PairwiseGameEvaluatorResult(
                qid=eval_tuple[0].qid,
                agent_a="new_agent",
                agent_b=eval_tuple[1].agent_b_answer.agent,
                evaluator_name="pairwise",
                answer=PairwiseEvaluationAnswer(
                    answer_a_analysis="A is good",
                    answer_b_analysis="B is bad",
                    comparison_reasoning="A is better",
                    winner="A",
                ),
            )

        mock_evaluator.evaluate_async = AsyncMock(side_effect=mock_evaluate_async)

        # Add the new agent's answers to the experiment queries so games can be played
        for qid, query in experiment.queries.items():
            query.answers["new_agent"] = AgentAnswer(qid=qid, agent="new_agent", text="New agent answer")

        result = await elo_ranker.add_agent_without_games(
            experiment=experiment,
            new_agent="new_agent",
            answer_evaluator=mock_evaluator,
        )

        assert "new_agent" in result.scores
        assert call_count > 0
        # New agent always wins → score should be above initial
        assert result.scores["new_agent"] >= elo_ranker.initial_score
