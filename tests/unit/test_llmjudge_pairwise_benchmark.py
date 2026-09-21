from pathlib import Path
from unittest.mock import AsyncMock

from ragelo import get_answer_evaluator
from ragelo.benchmarks.datasets import get_dataset
from ragelo.benchmarks.datasets.llmjudge_pairwise import PreferencePair
from ragelo.types.answer_formats import PairwiseEvaluationAnswer
from ragelo.types.formats import LLMResponseType

DATA_DIR = Path("tests/data/llmjudge")


def prefers(passage: str):
    """A judge that picks whichever assistant answered with `passage`, in either answer order."""

    def side_effect(prompt, schema):
        first, second = prompt.user_message.split("Assistant B]", 1)[0], prompt.user_message
        winner = "A" if passage in first else "B" if passage in second else "C"
        answer = PairwiseEvaluationAnswer(
            answer_a_analysis="a", answer_b_analysis="b", comparison_reasoning="c", winner=winner
        )
        return LLMResponseType(raw_answer=answer.model_dump_json(), parsed_answer=answer)

    return AsyncMock(side_effect=side_effect)


class TestPairwisePreferences:
    def test_pairs_put_the_higher_graded_passage_of_a_query_first(self):
        dataset = get_dataset("llmjudge_pairwise", data_dir=DATA_DIR, split="test", n_samples=10)

        assert dataset.pairs == [PreferencePair("q2", better="p2", worse="p3")]

    def test_accuracy_is_the_share_of_games_won_by_the_higher_graded_passage(self, llm_provider_mock, tmp_path):
        dataset = get_dataset("llmjudge_pairwise", data_dir=DATA_DIR, split="test", n_samples=10)
        evaluator = get_answer_evaluator("pairwise", llm_provider=llm_provider_mock)

        llm_provider_mock.async_call_mocker = prefers(dataset.data.passages["p2"])
        right = dataset.judge(evaluator, "right", tmp_path)
        llm_provider_mock.async_call_mocker = prefers(dataset.data.passages["p3"])
        wrong = dataset.judge(evaluator, "wrong", tmp_path)

        assert (right.n_games, right.accuracy, right.ties) == (1, 1.0, 0)
        assert (wrong.n_games, wrong.accuracy, wrong.ties) == (1, 0.0, 0)
        assert (right.n_failed, wrong.n_failed) == (0, 0)
        assert (right.points_by_grades, wrong.points_by_grades) == ({(2, 1): [1.0]}, {(2, 1): [0.0]})

    def test_a_tie_earns_half_a_point_in_every_breakdown(self, llm_provider_mock, tmp_path):
        dataset = get_dataset("llmjudge_pairwise", data_dir=DATA_DIR, split="test", n_samples=10)
        evaluator = get_answer_evaluator("pairwise", llm_provider=llm_provider_mock)
        llm_provider_mock.async_call_mocker = prefers("a passage nobody wrote")

        tied = dataset.judge(evaluator, "tied", tmp_path)

        assert (tied.accuracy, tied.accuracy_over_queries, tied.ties) == (0.5, 0.5, 1)
        assert (tied.points_by_grades, tied.points_by_query) == ({(2, 1): [0.5]}, {"q2": [0.5]})

    def test_a_game_with_a_failed_answer_order_counts_as_failed_and_leaves_no_accuracy(
        self, llm_provider_mock, tmp_path
    ):
        dataset = get_dataset("llmjudge_pairwise", data_dir=DATA_DIR, split="test", n_samples=10)
        evaluator = get_answer_evaluator("pairwise", llm_provider=llm_provider_mock)
        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=TimeoutError())

        found = dataset.judge(evaluator, "failed", tmp_path)

        assert (found.n_games, found.n_failed, found.accuracy) == (0, 1, None)
