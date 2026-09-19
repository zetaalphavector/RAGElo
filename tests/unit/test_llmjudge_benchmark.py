from collections import Counter
from pathlib import Path

import pytest
import typer

from benchmarks.agreement import agreement
from benchmarks.llmjudge import LLMJudgeData, load, sample, to_experiment
from benchmarks.run_llmjudge import EVALUATORS, Price, Row, judge
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
    @pytest.mark.parametrize("variant", [v for v, kwargs in EVALUATORS.items() if kwargs["evaluator_name"] != "jev"])
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
        found = agreement({"q": {"a": 0, "b": 2}}, {"q": {"a": 0, "b": 2}})

        assert Row(found, usages, Price(input=2, output=8, cached=0.5)).usage_cells() == ["200", "50", "50", "0.7250"]
        assert Row(found, usages, price=None).usage_cells() == ["200", "50", "50", "-"]
        assert Row(found, [], Price(input=2, output=8, cached=0.5)).usage_cells() == ["-", "-", "-", "-"]
