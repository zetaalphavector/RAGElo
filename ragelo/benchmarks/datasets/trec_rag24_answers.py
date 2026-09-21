"""TREC 2024 RAG, answer assessments: full system answers with NIST's nugget judgments.

A nugget is an atomic fact a good answer to the topic should contain, marked vital or okay. Every answer of a
topic is judged against all of the topic's nuggets as support, partial_support or not_support. That is the
shape of a RAGElo rubric, so a topic's nuggets become its rubric and each answer is judged against it.
"""

from __future__ import annotations

import json
import random
import statistics
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

from rich.console import RenderableType
from rich.table import Table

from ragelo import Experiment, get_agent_ranker
from ragelo.benchmarks.agreement import kendall_tau, spearman
from ragelo.benchmarks.datasets.base import DatasetFactory
from ragelo.benchmarks.datasets.pairwise import PairwiseDataset, game_usage
from ragelo.benchmarks.pricing import Price, usage_cells
from ragelo.benchmarks.throughput import Throughput
from ragelo.evaluators.answer_evaluators import BaseAnswerEvaluator
from ragelo.types.answer_formats import Criterion
from ragelo.types.configurations import TrecRag24AnswersDatasetConfig
from ragelo.types.formats import LLMUsage
from ragelo.types.types import BenchmarkDatasetTypes

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


def human_scores(data: AnswerData, systems: list[str] | None = None) -> dict[str, float]:
    """Mean vital strict score per system, the track's primary measure, over the topics it answered."""
    per_system: dict[str, list[float]] = {}
    for qid, by_run in data.answers.items():
        for run in by_run:
            if systems is None or run in systems:
                score = nugget_score(data.nuggets[qid], human_fulfillments(data, qid, run))
                per_system.setdefault(run, []).append(score)
    return {run: statistics.mean(scores) for run, scores in per_system.items()}


def pick_systems(data: AnswerData, n_systems: int, seed: int = 42) -> list[str]:
    """The best and the worst system by the human scores, and one system drawn at random from each of
    `n_systems - 2` equal bands of the ranks between them, among the systems that answered every topic."""
    complete = [run for run in human_scores(data) if all(run in by_run for by_run in data.answers.values())]
    scores = human_scores(data, complete)
    ranked = sorted(scores, key=lambda run: (-scores[run], run))
    between, n_bands, rng = ranked[1:-1], n_systems - 2, random.Random(seed)
    drawn = [
        rng.choice(between[band * len(between) // n_bands : (band + 1) * len(between) // n_bands])
        for band in range(n_bands)
    ]
    return [ranked[0], *drawn, ranked[-1]]


def keep_systems(data: AnswerData, systems: list[str]) -> AnswerData:
    return replace(
        data,
        answers={
            qid: {run: text for run, text in by_run.items() if run in systems} for qid, by_run in data.answers.items()
        },
        assignments={
            qid: {run: labels for run, labels in by_run.items() if run in systems}
            for qid, by_run in data.assignments.items()
        },
    )


@dataclass(frozen=True, slots=True)
class Tournament:
    """`elo` is RAGElo's rating per system. A game agrees when its winner is the answer with the higher
    human score; games between answers with equal human scores are left out of that count."""

    elo: dict[str, float]
    wins: dict[str, int]
    n_games: int
    n_failed: int
    decided: int
    agreeing: int
    ties: int
    usages: list[LLMUsage]
    throughput: Throughput


@DatasetFactory.register(BenchmarkDatasetTypes.TREC_RAG24_ANSWERS)
class TrecRag24AnswersDataset(PairwiseDataset[Tournament]):
    """An Elo tournament between `n_systems` of the track's systems, against their ranking by NIST's nugget
    scores. `n_samples` is a difficulty-stratified sample of topics."""

    config: TrecRag24AnswersDatasetConfig
    data_name = "trec_rag24_answers"

    def download(self) -> None:
        download(self.data_dir)

    @cached_property
    def data(self) -> AnswerData:
        everything = load(self.data_dir)
        systems = pick_systems(everything, self.config.n_systems, self.config.seed)
        topics = (
            everything
            if self.config.n_samples is None
            else sample_topics(everything, self.config.n_samples, self.config.seed)
        )
        return keep_systems(topics, systems)

    @property
    def subset(self) -> str:
        return f"{len(self.data.queries)}topics_{self.config.n_systems}systems_seed{self.config.seed}"

    def evaluator_options(self, variant: str) -> dict[str, Any]:
        return {"rubrics": rubrics(self.data)} if "rubric" in variant else {}

    def to_experiment(self, experiment_name: str, save_path: str | None = None) -> Experiment:
        experiment = Experiment(experiment_name, save_path=save_path)
        for qid, text in self.data.queries.items():
            experiment.add_query(text, query_id=qid, should_save=False)
            for run_id, answer in self.data.answers[qid].items():
                experiment.add_agent_answer(answer, run_id, qid, exist_ok=True, should_save=False)
        experiment.save()
        return experiment

    def outcome(self, experiment: Experiment, evaluator: BaseAnswerEvaluator, throughput: Throughput) -> Tournament:
        name = str(evaluator.config.evaluator_name)
        n_games = decided = agreeing = ties = 0
        usages = []
        for query in experiment:
            human = {
                run: nugget_score(self.data.nuggets[query.qid], human_fulfillments(self.data, query.qid, run))
                for run in self.data.answers[query.qid]
            }
            for game in query.pairwise_games.values():
                result = game.evaluations.get(name)
                if result is None or result.answer is None:
                    continue
                n_games += 1
                if usage := game_usage(result):
                    usages.append(usage)
                agent_a, agent_b = game.agent_a_answer.agent, game.agent_b_answer.agent
                winner = {"A": agent_a, "B": agent_b}.get(result.answer.winner)
                ties += winner is None
                if human[agent_a] != human[agent_b] and winner is not None:
                    decided += 1
                    agreeing += winner == max((agent_a, agent_b), key=human.__getitem__)

        # The ranker shuffles the games of each of its tournaments with the global generator.
        random.seed(self.config.seed)
        ranking = get_agent_ranker("elo").run(experiment)
        return Tournament(
            elo=ranking.scores,
            # The ranker adds up the wins of every tournament it replays the same games in.
            wins={run: wins // ranking.total_tournaments for run, wins in ranking.wins.items()},
            n_games=n_games,
            n_failed=len(self.items(experiment)) - n_games,
            decided=decided,
            agreeing=agreeing,
            ties=ties,
            usages=usages,
            throughput=throughput,
        )

    def tables(
        self, outcomes: Mapping[tuple[str, str], Tournament], prices: Mapping[str, Price]
    ) -> list[RenderableType]:
        human = human_scores(self.data)
        by_human = sorted(human, key=lambda run: -human[run])
        scores = [human[run] for run in by_human]
        rendered: list[RenderableType] = []
        for (name, model), played in outcomes.items():
            table = Table(title=f"{name} on {model}: {len(human)} systems, {len(self.data.queries)} topics")
            for column in ("system", "human vital strict", "human rank", "Elo", "Elo rank", "games won"):
                table.add_column(column)
            by_elo = sorted(human, key=lambda run: -played.elo.get(run, 0.0))
            for run in by_human:
                table.add_row(
                    run,
                    f"{human[run]:.3f}",
                    str(by_human.index(run) + 1),
                    f"{played.elo.get(run, float('nan')):.0f}",
                    str(by_elo.index(run) + 1),
                    str(played.wins.get(run, 0)),
                )
            ratings = [played.elo.get(run, 0.0) for run in by_human]
            tokens = usage_cells(played.usages, played.n_games, prices.get(model))
            agreement = f"{played.agreeing / played.decided:.1%}" if played.decided else "-"
            cost = f", {tokens[3]} USD per 1,000 games" if model in prices else ""
            summary = (
                f"Kendall tau {kendall_tau(scores, ratings):.3f} and Spearman "
                f"{spearman(scores, ratings):.3f} between the Elo ratings and the human scores.\n"
                f"Games: {played.n_games} judged, {played.n_failed} failed, {played.ties} tied. The winner is the "
                f"answer with the higher human score in {played.agreeing} of the {played.decided} games that have "
                f"one ({agreement}).\n"
                f"Per game: {tokens[0]} input and {tokens[2]} output tokens{cost}, "
                f"{played.throughput.cell()} games per second.\n"
            )
            rendered += [table, summary]
        return rendered
