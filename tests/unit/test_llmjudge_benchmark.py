from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import typer

from benchmarks.agreement import spearman_interval
from benchmarks.llmjudge import LLMJudgeData, load, sample, to_experiment
from benchmarks.run_llmjudge import EVALUATORS, Judgments, Price, common_pairs, judge, rank, usage_cells
from benchmarks.throughput import Run, Throughput, record
from ragelo import get_retrieval_evaluator
from ragelo.types.formats import LLMUsage

DATA_DIR = Path("tests/data/llmjudge")


class TestLLMJudgeLoader:
    @pytest.mark.parametrize(
        "split,expected_qrels",
        [
            ("dev", {"q1": {"p1": 3, "p3": 0}}),
            ("test", {"q2": {"p2": 2, "p3": 1}}),
        ],
    )
    def test_load_keeps_only_what_the_split_judged(self, split, expected_qrels):
        data = load(DATA_DIR, split)

        assert data.qrels == expected_qrels
        assert data.queries.keys() == expected_qrels.keys()
        assert data.passages.keys() == {did for labels in expected_qrels.values() for did in labels}
        assert data.n_pairs == 2

    def test_experiment_holds_the_judged_pairs_without_the_human_labels(self, tmp_path):
        data = load(DATA_DIR, "dev")

        experiment = to_experiment(data, "llmjudge_dev", save_path=str(tmp_path / "llmjudge_dev.json"))

        pairs = {(query.qid, doc.did) for query in experiment for doc in query.retrieved_docs_iter()}
        assert pairs == {("q1", "p1"), ("q1", "p3")}
        assert experiment["q1"].query == "dog age by teeth"
        assert experiment["q1"].retrieved_docs["p1"].text == "Puppies get adult teeth at six months."
        assert all(doc.metadata is None for query in experiment for doc in query.retrieved_docs_iter())

    def test_sample_keeps_the_label_distribution(self):
        labels = {f"p{i}": 0 if i < 30 else 3 for i in range(40)}
        data = LLMJudgeData(queries={"q1": "query"}, passages={did: did for did in labels}, qrels={"q1": labels})

        subset = sample(data, n_pairs=20)

        assert Counter(subset.qrels["q1"].values()) == {0: 15, 3: 5}
        assert subset.passages.keys() == subset.qrels["q1"].keys()
        assert sample(data, n_pairs=20).qrels == subset.qrels


class TestLLMJudgeRunner:
    # The jev variants only accept the vercel-jev provider, which tests/unit/test_jev.py covers.
    @pytest.mark.parametrize(
        "variant", [v for v, kwargs in EVALUATORS.items() if not kwargs["evaluator_name"].startswith("jev")]
    )
    def test_every_variant_in_the_grid_judges_through_an_experiment(self, variant, llm_provider_mock, tmp_path):
        data = load(DATA_DIR, "test")
        evaluator = get_retrieval_evaluator(llm_provider=llm_provider_mock, **EVALUATORS[variant])

        judgments = judge(data, evaluator, f"test_{variant}_mock", tmp_path)

        assert judgments.scores["q2"].keys() == data.qrels["q2"].keys()

    def test_judge_labels_every_pair_and_a_rerun_makes_no_llm_calls(self, llm_provider_mock_retrieval, tmp_path):
        data = load(DATA_DIR, "test")
        evaluator = get_retrieval_evaluator("reasoner", llm_provider=llm_provider_mock_retrieval)

        scores = judge(data, evaluator, "test_reasoner_mock", tmp_path).scores

        assert scores == {"q2": {"p2": 2, "p3": 2}}
        assert llm_provider_mock_retrieval.async_call_mocker.call_count == 2

        assert judge(data, evaluator, "test_reasoner_mock", tmp_path).scores == scores
        assert llm_provider_mock_retrieval.async_call_mocker.call_count == 2

    def test_a_failed_evaluation_is_counted_and_left_out_of_the_scores(self, llm_provider_mock_retrieval, tmp_path):
        data = load(DATA_DIR, "test")
        answer = llm_provider_mock_retrieval.async_call_mocker.side_effect

        def fail_on_p3(prompt, schema):
            if data.passages["p3"] in prompt.user_message:
                raise TimeoutError()
            return answer(prompt, schema)

        llm_provider_mock_retrieval.async_call_mocker = AsyncMock(side_effect=fail_on_p3)
        evaluator = get_retrieval_evaluator("reasoner", llm_provider=llm_provider_mock_retrieval)

        judgments = judge(data, evaluator, "one_failure", tmp_path)

        assert (judgments.scores, judgments.n_failed) == ({"q2": {"p2": 2}}, 1)


class TestThroughput:
    def test_runs_stay_apart_and_the_median_is_taken_at_the_latest_parallelism(self, tmp_path):
        path = tmp_path / "throughput.json"
        record(path, Run(evaluations=500, seconds=10.0, n_processes=32))
        record(path, Run(evaluations=500, seconds=25.0, n_processes=16))
        record(path, Run(evaluations=500, seconds=20.0, n_processes=16))
        record(path, Run(evaluations=2, seconds=60.0, n_processes=16))

        after_a_cached_run = record(path, Run(evaluations=0, seconds=0.1, n_processes=16))

        assert [run.per_second for run in after_a_cached_run.runs] == [50.0, 20.0, 25.0]
        assert after_a_cached_run.cell() == "22.5 [20.0-25.0] (x16, 2 runs)"

    def test_the_parallelism_names_the_calls_behind_one_evaluation(self):
        five_annotators = Throughput((Run(evaluations=500, seconds=50.0, n_processes=16, calls_per_evaluation=5),))

        assert five_annotators.cell() == "10.0 (x16, 5 calls each, 1 run)"
        assert Throughput().cell() == "-"


def judgments(scores: dict[str, dict[str, float]]) -> Judgments:
    return Judgments(scores=scores, usages=[], n_failed=0, throughput=Throughput())


class TestCommonPairs:
    def test_rows_are_compared_on_the_pairs_every_judge_scored(self):
        judged = [judgments({"q": {"a": 2, "b": 1}}), judgments({"q": {"a": 0}}), judgments({})]
        assert common_pairs(judged) == {("q", "a")}, "a judge that scored nothing must not empty every row"

    def test_no_judge_scored_anything(self):
        assert common_pairs([judgments({})]) == set()


class TestRanking:
    def test_a_judge_without_a_defined_spearman_ranks_last(self):
        assert sorted([0.5, float("nan"), 0.9, float("nan"), 0.7], key=rank)[:3] == [0.9, 0.7, 0.5]

    def test_the_interval_resamples_queries_and_is_wider_than_one_over_pairs_suggests(self):
        """Two queries the judge ranks perfectly and two it ranks backwards: which ones a resample
        draws decides the estimate, so the interval spans both signs."""
        human = {f"q{i}": {"a": 0, "b": 1, "c": 2} for i in range(4)}
        judged = {
            qid: dict(labels) if qid in ("q0", "q1") else {"a": 2, "b": 1, "c": 0} for qid, labels in human.items()
        }

        low, high = spearman_interval(human, judged, n_resamples=200)

        assert low < 0 < high
        assert spearman_interval(human, human, n_resamples=50) == (1.0, 1.0)


class TestCost:
    def test_cached_input_tokens_are_billed_at_the_cached_rate(self):
        model, price = Price.parse("openai/gpt-x=2,8,0.5")

        assert model == "openai/gpt-x"
        assert price.cost(LLMUsage(input_tokens=1_000_000, output_tokens=500_000, cached_tokens=400_000)) == (
            pytest.approx(0.6 * 2 + 0.4 * 0.5 + 0.5 * 8)
        )

    def test_a_price_needs_a_model_and_three_rates(self):
        with pytest.raises(typer.BadParameter):
            Price.parse("gpt-x=2,8")

    def test_the_table_reports_mean_tokens_and_the_cost_of_a_thousand_pairs(self):
        usages = [
            LLMUsage(input_tokens=300, output_tokens=50, cached_tokens=100),
            LLMUsage(input_tokens=100, output_tokens=50),
        ]
        price = Price(input=2, output=8, cached=0.5)

        assert usage_cells(usages, 2, price) == ["200", "50", "50", "0.7250"]
        assert usage_cells(usages, 2, price=None) == ["200", "50", "50", "-"]
        assert usage_cells(usages[:1], 2, price) == ["-", "-", "-", "-"]
