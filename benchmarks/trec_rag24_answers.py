"""TREC 2024 RAG, answer assessments: full system answers with NIST's nugget judgments.

A nugget is an atomic fact a good answer to the topic should contain, marked vital or okay. Every answer of a
topic is judged against all of the topic's nuggets as support, partial_support or not_support. That is the
shape of a RAGElo rubric, so a topic's nuggets become its rubric and each answer is judged against it.
"""

from __future__ import annotations

import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ragelo import Experiment
from ragelo.types.answer_formats import Criterion

NUGGETS_URL = "https://trec.nist.gov/data/rag/nugget_assignment.20241218.jsonl"
NUGGETS_FILE = "nugget_assignment.20241218.jsonl"
# Scoring of the track (Pradeep et al., arXiv 2411.09607, section 3.5). Vital strict is its primary score.
SUPPORT = {"support": 1.0, "partial_support": 0.5, "not_support": 0.0}
WEIGHT = {"vital": 1.0, "okay": 0.5}

Importance = Literal["vital", "okay"]
Assignment = Literal["support", "partial_support", "not_support"]


@dataclass(frozen=True, slots=True)
class Nugget:
    name: str
    text: str
    importance: Importance


@dataclass(frozen=True, slots=True)
class AnswerData:
    """`assignments[qid][run_id]` has one label per nugget of `nuggets[qid]`, in the same order."""

    queries: dict[str, str]
    nuggets: dict[str, list[Nugget]]
    answers: dict[str, dict[str, str]]
    assignments: dict[str, dict[str, list[Assignment]]]

    @property
    def n_answers(self) -> int:
        return sum(len(by_run) for by_run in self.answers.values())


def download(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / NUGGETS_FILE
    if not target.is_file():
        # trec.nist.gov answers 403 to urllib's default user agent.
        request = urllib.request.Request(NUGGETS_URL, headers={"User-Agent": "ragelo-benchmarks"})
        target.write_bytes(urllib.request.urlopen(request, timeout=120).read())


def load(data_dir: Path) -> AnswerData:
    queries: dict[str, str] = {}
    nuggets: dict[str, list[Nugget]] = {}
    answers: dict[str, dict[str, str]] = {}
    assignments: dict[str, dict[str, list[Assignment]]] = {}
    for line in (data_dir / NUGGETS_FILE).read_text().splitlines():
        row = json.loads(line)
        qid = row["qid"]
        topic_nuggets = [
            Nugget(f"nugget_{i:02d}", nugget["text"].strip(), nugget["importance"])
            for i, nugget in enumerate(row["nuggets"], start=1)
        ]
        if nuggets.setdefault(qid, topic_nuggets) != topic_nuggets:
            raise ValueError(f"Run {row['run_id']} was judged against other nuggets than the rest of topic {qid}")
        queries[qid] = row["query"]
        answers.setdefault(qid, {})[row["run_id"]] = row["answer_text"]
        assignments.setdefault(qid, {})[row["run_id"]] = [nugget["assignment"] for nugget in row["nuggets"]]
    return AnswerData(queries=queries, nuggets=nuggets, answers=answers, assignments=assignments)


def rubrics(data: AnswerData) -> dict[str, list[Criterion]]:
    """One rubric per topic, for the `rubrics` option of the rubric evaluators.

    With the track's weights, 1 for vital and 0.5 for okay, a graduated judgment on a 0-2 scale gives
    `average_score` the value of the track's weighted score.
    """
    return {
        qid: [
            Criterion(
                criterion_name=nugget.name,
                short_question=f"Does the report state that: {nugget.text}?",
                weight=WEIGHT[nugget.importance],
            )
            for nugget in topic_nuggets
        ]
        for qid, topic_nuggets in data.nuggets.items()
    }


def nugget_score(
    nuggets: list[Nugget], fulfillments: list[float], vital_only: bool = True, strict: bool = True
) -> float:
    """The track's scores of one answer, from a fulfillment per nugget between 0 and 1.

    Vital strict by default. `vital_only=False` is the weighted score, `strict=False` counts partial support.
    """
    weights = [1.0 if vital_only else WEIGHT[nugget.importance] for nugget in nuggets]
    kept = [
        (weight, float(fulfillment == 1.0) if strict else fulfillment)
        for nugget, weight, fulfillment in zip(nuggets, weights, fulfillments, strict=True)
        if nugget.importance == "vital" or not vital_only
    ]
    total = sum(weight for weight, _ in kept)
    return sum(weight * fulfillment for weight, fulfillment in kept) / total if total else 0.0


def human_fulfillments(data: AnswerData, qid: str, run_id: str) -> list[float]:
    return [SUPPORT[assignment] for assignment in data.assignments[qid][run_id]]


def sample_topics(data: AnswerData, n_topics: int, seed: int = 42, n_strata: int = 4) -> AnswerData:
    """Draws topics evenly from strata of difficulty, the mean vital strict score of a topic's answers.

    This keeps easy and hard topics in every sample. It does not make the system ranking more stable than
    a simple random draw: against the ranking on all 56 topics both reach a Kendall tau of 0.81 at 20 topics.
    """
    difficulty = {
        qid: sum(nugget_score(data.nuggets[qid], human_fulfillments(data, qid, run)) for run in runs) / len(runs)
        for qid, runs in data.answers.items()
    }
    ordered = sorted(difficulty, key=lambda qid: (difficulty[qid], qid))
    rng = random.Random(seed)
    kept: list[str] = []
    for i in range(n_strata):
        stratum = ordered[i * len(ordered) // n_strata : (i + 1) * len(ordered) // n_strata]
        kept += rng.sample(stratum, min(len(stratum), n_topics // n_strata + (i < n_topics % n_strata)))
    return AnswerData(
        queries={qid: data.queries[qid] for qid in kept},
        nuggets={qid: data.nuggets[qid] for qid in kept},
        answers={qid: data.answers[qid] for qid in kept},
        assignments={qid: data.assignments[qid] for qid in kept},
    )


def to_experiment(data: AnswerData, experiment_name: str, save_path: str | None = None) -> Experiment:
    """Every run is an agent. The nuggets and their labels stay in `data`, out of reach of any prompt."""
    experiment = Experiment(experiment_name, save_path=save_path)
    for qid, text in data.queries.items():
        experiment.add_query(text, query_id=qid, should_save=False)
        for run_id, answer in data.answers[qid].items():
            experiment.add_agent_answer(answer, run_id, qid, exist_ok=True, should_save=False)
    experiment.save()
    return experiment
