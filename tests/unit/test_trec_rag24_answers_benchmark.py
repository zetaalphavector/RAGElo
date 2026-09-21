import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ragelo import get_answer_evaluator
from ragelo.benchmarks.datasets import get_dataset
from ragelo.benchmarks.datasets.trec_rag24_answers import (
    NUGGETS_FILE,
    Nugget,
    human_fulfillments,
    load,
    nugget_score,
    rubrics,
    sample_topics,
)
from ragelo.types.formats import LLMResponseType

NUGGETS = [("Paris is the capital", "vital"), ("It has two million people", "vital"), ("It is on the Seine", "okay")]


def write_rows(data_dir: Path, runs: dict[str, list[str]], nuggets=NUGGETS) -> None:
    rows = [
        {
            "qid": "2024-1",
            "query": "what is the capital of France?",
            "run_id": run_id,
            "answer_text": f"Answer of {run_id}",
            "response_length": 3,
            "nuggets": [
                {"text": text, "importance": importance, "assignment": label}
                for (text, importance), label in zip(nuggets, labels, strict=True)
            ],
        }
        for run_id, labels in runs.items()
    ]
    (data_dir / NUGGETS_FILE).write_text("\n".join(json.dumps(row) for row in rows))


class TestNuggetsAsRubrics:
    def test_a_topics_nuggets_become_one_weighted_rubric_shared_by_its_answers(self, tmp_path):
        write_rows(tmp_path, {"run_a": ["support", "partial_support", "not_support"], "run_b": ["support"] * 3})

        data = load(tmp_path)
        rubric = rubrics(data)["2024-1"]

        assert data.n_answers == 2
        assert [(c.criterion_name, c.weight) for c in rubric] == [
            ("nugget_01", 1.0),
            ("nugget_02", 1.0),
            ("nugget_03", 0.5),
        ]
        assert rubric[0].short_question == "Does the report state that: Paris is the capital?"
        assert human_fulfillments(data, "2024-1", "run_a") == [1.0, 0.5, 0.0]

    @pytest.mark.parametrize(
        ("vital_only", "strict", "expected"),
        [(True, True, 0.5), (True, False, 0.75), (False, True, 1.5 / 2.5), (False, False, 2.0 / 2.5)],
    )
    def test_nugget_score_follows_the_tracks_vital_weighted_and_strict_scores(self, vital_only, strict, expected):
        """support, partial_support and support on a vital, a vital and an okay nugget."""
        data_nuggets = load_nuggets()

        assert nugget_score(data_nuggets, [1.0, 0.5, 1.0], vital_only=vital_only, strict=strict) == pytest.approx(
            expected
        )

    def test_a_run_judged_against_other_nuggets_is_an_error(self, tmp_path):
        write_rows(tmp_path, {"run_a": ["support"] * 3})
        other = json.loads((tmp_path / NUGGETS_FILE).read_text())
        other["run_id"] = "run_b"
        other["nuggets"][0]["text"] = "Lyon is the capital"
        with (tmp_path / NUGGETS_FILE).open("a") as f:
            f.write("\n" + json.dumps(other))

        with pytest.raises(ValueError, match="other nuggets"):
            load(tmp_path)

    def test_a_graduated_rubric_judgment_reproduces_the_weighted_score_of_the_same_labels(
        self, llm_provider_mock, tmp_path
    ):
        """Judging on a 0-2 scale with the track's weights makes `average_score` the track's weighted score."""
        write_rows(tmp_path, {"run_a": ["support", "partial_support", "not_support"]})
        dataset = get_dataset("trec_rag24_answers", data_dir=tmp_path, n_systems=2)
        data = dataset.data
        grades = {"nugget_01": 2, "nugget_02": 1, "nugget_03": 0}
        llm_provider_mock.async_call_mocker = AsyncMock(
            side_effect=lambda prompt, schema: LLMResponseType(
                raw_answer="{}",
                parsed_answer=schema(**{name: {"reasoning": "r", "score": grade} for name, grade in grades.items()}),
            )
        )
        evaluator = get_answer_evaluator(
            "rubric_pointwise",
            llm_provider=llm_provider_mock,
            rubrics=rubrics(data),
            graduated_scoring=True,
            max_score=2,
        )
        experiment = dataset.to_experiment("answers", save_path=str(tmp_path / "answers.json"))

        evaluator.evaluate_experiment(experiment)

        judged = experiment["2024-1"].answers["run_a"].evaluations["rubric_pointwise"].answer
        human = nugget_score(
            data.nuggets["2024-1"], human_fulfillments(data, "2024-1", "run_a"), vital_only=False, strict=False
        )
        assert [c.fulfillment for c in judged.criteria] == [1.0, 0.5, 0.0]
        assert judged.average_score == pytest.approx(human) == pytest.approx(1.5 / 2.5)
        prompt = llm_provider_mock.async_call_mocker.call_args_list[0][0][0]
        assert "Does the report state that: Paris is the capital?" in prompt.system_prompt
        assert "Supporting Documents" not in prompt.system_prompt


def load_nuggets():
    return [Nugget(f"nugget_{i:02d}", text, importance) for i, (text, importance) in enumerate(NUGGETS, start=1)]


class TestTopicSample:
    def test_topics_are_drawn_from_every_difficulty_stratum_with_all_of_their_answers(self, tmp_path):
        rows = []
        for i in range(8):
            labels = ["support"] * (i // 2) + ["not_support"] * (3 - i // 2)
            rows.append(
                {
                    "qid": f"2024-{i}",
                    "query": f"query {i}",
                    "run_id": "run_a",
                    "answer_text": "answer",
                    "nuggets": [
                        {"text": f"fact {j}", "importance": "vital", "assignment": label}
                        for j, label in enumerate(labels)
                    ],
                }
            )
        (tmp_path / NUGGETS_FILE).write_text("\n".join(json.dumps(row) for row in rows))
        data = load(tmp_path)

        sampled = sample_topics(data, n_topics=4, seed=1)

        hardness = sorted(int(qid.split("-")[1]) // 2 for qid in sampled.queries)
        assert hardness == [0, 1, 2, 3]
        assert sampled.answers.keys() == sampled.nuggets.keys() == sampled.assignments.keys() == sampled.queries.keys()
        assert sample_topics(data, n_topics=4, seed=1).queries == sampled.queries
