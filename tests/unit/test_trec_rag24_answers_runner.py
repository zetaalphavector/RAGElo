import json
from pathlib import Path
from unittest.mock import AsyncMock

from ragelo import get_answer_evaluator
from ragelo.benchmarks.datasets import get_dataset
from ragelo.benchmarks.datasets.trec_rag24_answers import NUGGETS_FILE, human_scores, load, pick_systems
from ragelo.types.answer_formats import PairwiseEvaluationAnswer
from ragelo.types.formats import LLMResponseType

# Twelve systems, system i supporting i of 11 vital nuggets on both topics, so their human ranking is strict.
SYSTEMS = [f"system_{i:02d}" for i in range(12)]


def write_answers(data_dir: Path) -> None:
    rows = [
        {
            "qid": qid,
            "query": f"query {qid}",
            "run_id": system,
            "answer_text": f"{system} answers {qid}",
            "nuggets": [
                {"text": f"fact {n}", "importance": "vital", "assignment": "support" if n < i else "not_support"}
                for n in range(11)
            ],
        }
        for qid in ("2024-1", "2024-2")
        for i, system in enumerate(SYSTEMS)
    ]
    (data_dir / NUGGETS_FILE).write_text("\n".join(json.dumps(row) for row in rows))


def prefers_the_higher_numbered_system(prompt, schema):
    first, second = prompt.user_message.split("Assistant B]", 1)[0], prompt.user_message
    in_a = max(system for system in SYSTEMS if system in first)
    in_b = max(system for system in SYSTEMS if system in second.split("Assistant B]", 1)[1])
    answer = PairwiseEvaluationAnswer(
        answer_a_analysis="a", answer_b_analysis="b", comparison_reasoning="c", winner="A" if in_a > in_b else "B"
    )
    return LLMResponseType(raw_answer=answer.model_dump_json(), parsed_answer=answer)


class TestSystemRanking:
    def test_systems_come_from_the_top_the_bottom_and_every_band_between(self, tmp_path):
        write_answers(tmp_path)
        data = load(tmp_path)

        picked = pick_systems(data, n_systems=4, seed=1)

        assert (picked[0], picked[-1]) == ("system_11", "system_00")
        assert picked[1] in SYSTEMS[6:11] and picked[2] in SYSTEMS[1:6]
        assert pick_systems(data, n_systems=4, seed=1) == picked

    def test_a_judge_that_follows_the_human_scores_ranks_the_systems_like_the_assessors(
        self, llm_provider_mock, tmp_path
    ):
        write_answers(tmp_path)
        dataset = get_dataset("trec_rag24_answers", data_dir=tmp_path, n_systems=4)
        human = human_scores(dataset.data)
        systems = sorted(human, reverse=True)
        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=prefers_the_higher_numbered_system)
        evaluator = get_answer_evaluator("pairwise", llm_provider=llm_provider_mock)

        played = dataset.judge(evaluator, "elo", tmp_path)

        assert (played.n_games, played.n_failed, played.ties) == (12, 0, 0)
        assert (played.agreeing, played.decided) == (12, 12)
        assert (systems[0], systems[-1]) == ("system_11", "system_00")
        assert sorted(systems, key=lambda run: -played.elo[run]) == systems
        assert sorted(systems, key=lambda run: -human[run]) == systems
