from ragelo.presenters import render_evaluation, render_failed_evaluations
from ragelo.types.results import (
    AnswerEvaluationAnswer,
    AnswerEvaluatorResult,
    EloTournamentResult,
    PairwiseEvaluationAnswer,
    PairwiseGameEvaluatorResult,
    RetrievalEvaluationAnswer,
    RetrievalEvaluatorResult,
)


class TestRenderEvaluation:
    def test_render_retrieval(self, capsys):
        result = RetrievalEvaluatorResult(
            qid="q1",
            did="d1",
            evaluator_name="reasoner",
            answer=RetrievalEvaluationAnswer(reasoning="Good doc", score=2),
        )
        render_evaluation(result, rich_print=False)
        out = capsys.readouterr().out
        assert "Query ID: q1" in out
        assert "Document ID: d1" in out

    def test_render_pairwise(self, capsys):
        result = PairwiseGameEvaluatorResult(
            qid="q1",
            agent_a="a1",
            agent_b="a2",
            evaluator_name="pairwise",
            answer=PairwiseEvaluationAnswer(
                answer_a_analysis="Fine",
                answer_b_analysis="Better",
                comparison_reasoning="B wins",
                winner="B",
            ),
        )
        render_evaluation(result, rich_print=False)
        out = capsys.readouterr().out
        assert "Agent A: a1" in out
        assert "Agent B: a2" in out

    def test_render_answer(self, capsys):
        result = AnswerEvaluatorResult(
            qid="q1",
            agent="a1",
            evaluator_name="test",
            answer=AnswerEvaluationAnswer(reasoning="Great", score=5),
        )
        render_evaluation(result, rich_print=False)
        out = capsys.readouterr().out
        assert "Query ID: q1" in out
        assert "Agent: a1" in out

    def test_render_elo_tournament(self, capsys):
        result = EloTournamentResult(
            agents=["a", "b"],
            scores={"a": 1050.0, "b": 950.0},
            games_played={"a": 10, "b": 10},
            wins={"a": 6, "b": 4},
            loses={"a": 4, "b": 6},
            ties={"a": 0, "b": 0},
            std_dev={"a": 12.0, "b": 15.0},
            total_games=10,
            total_tournaments=1,
        )
        render_evaluation(result, rich_print=False)
        out = capsys.readouterr().out
        assert "Agents Elo Ratings" in out
        assert "1050.0" in out


class TestRenderFailedEvaluations:
    def test_no_failures(self, capsys):
        render_failed_evaluations(total_evaluations=10, failed_evaluations=0, rich_print=False)
        out = capsys.readouterr().out
        assert "Done!" in out
        assert "Total evaluations: 10" in out
        assert "Failed" not in out

    def test_with_failures(self, capsys):
        render_failed_evaluations(total_evaluations=10, failed_evaluations=3, rich_print=False)
        out = capsys.readouterr().out
        assert "Failed evaluations: 3" in out
        assert "Total evaluations: 10" in out
