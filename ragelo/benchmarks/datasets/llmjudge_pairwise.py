"""Pairwise preference on LLMJudge: two passages of one query with different human labels stand in for two
agents' answers, and the judge should prefer the one the assessors graded higher."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from statistics import fmean

from rich.console import RenderableType
from rich.table import Table

from ragelo import Experiment
from ragelo.benchmarks.datasets.base import DatasetFactory
from ragelo.benchmarks.datasets.llmjudge import download, load
from ragelo.benchmarks.datasets.pairwise import PairwiseDataset, game_usage
from ragelo.benchmarks.datasets.retrieval import RetrievalData
from ragelo.benchmarks.pricing import Price, usage_cells
from ragelo.benchmarks.throughput import Throughput
from ragelo.evaluators.answer_evaluators import BaseAnswerEvaluator
from ragelo.types.formats import LLMUsage
from ragelo.types.types import BenchmarkDatasetTypes

MAX_PASSAGE_CHARS = 5_000
N_PAIRS = 500


@dataclass(frozen=True, slots=True)
class PreferencePair:
    qid: str
    better: str
    worse: str


@dataclass(frozen=True, slots=True)
class Preferences:
    """A tie earns half a point, so a judge that always ties scores 0.5 like one that guesses."""

    n_games: int
    n_failed: int
    correct: int
    ties: int
    usages: list[LLMUsage]
    throughput: Throughput
    points_by_grades: dict[tuple[int, int], list[float]]
    points_by_query: dict[str, list[float]]

    @property
    def accuracy(self) -> float | None:
        """None when no game was judged. A game with a failed answer order is not judged."""
        return (self.correct + self.ties / 2) / self.n_games if self.n_games else None

    @property
    def accuracy_over_queries(self) -> float | None:
        """The mean of the per-query accuracies, so a query with many sampled pairs counts once."""
        if not self.points_by_query:
            return None
        return fmean(fmean(points) for points in self.points_by_query.values())


def sample_pairs(data: RetrievalData, n_pairs: int, seed: int = 42) -> list[PreferencePair]:
    rng = random.Random(seed)
    candidates = [
        PreferencePair(qid, better, worse)
        for qid, labels in sorted(data.qrels.items())
        for better in sorted(labels)
        for worse in sorted(labels)
        if labels[better] > labels[worse]
        and max(len(data.passages[better]), len(data.passages[worse])) <= MAX_PASSAGE_CHARS
    ]
    return rng.sample(candidates, min(n_pairs, len(candidates)))


@DatasetFactory.register(BenchmarkDatasetTypes.LLMJUDGE_PAIRWISE)
class LLMJudgePairwiseDataset(PairwiseDataset[Preferences]):
    data_name = "llmjudge"
    splits = ("dev", "test")

    def download(self) -> None:
        download(self.data_dir)

    @cached_property
    def data(self) -> RetrievalData:
        return load(self.data_dir, self.split)

    @cached_property
    def pairs(self) -> list[PreferencePair]:
        return sample_pairs(self.data, self.config.n_samples or N_PAIRS, self.config.seed)

    @property
    def subset(self) -> str:
        return f"{self.split}_{self.config.n_samples or N_PAIRS}_seed{self.config.seed}"

    def to_experiment(self, experiment_name: str, save_path: str | None = None) -> Experiment:
        """One query per pair. The better passage is agent `a` or `b` at random, so its name gives nothing away."""
        rng = random.Random(experiment_name)
        experiment = Experiment(experiment_name, save_path=save_path)
        for i, pair in enumerate(self.pairs):
            qid = f"{pair.qid}_{i}"
            better_agent, worse_agent = rng.sample(["a", "b"], 2)
            experiment.add_query(
                self.data.queries[pair.qid], query_id=qid, metadata={"better": better_agent}, should_save=False
            )
            for passage, agent in ((pair.better, better_agent), (pair.worse, worse_agent)):
                experiment.add_agent_answer(self.data.passages[passage], agent, qid, exist_ok=True, should_save=False)
        experiment.save()
        return experiment

    def outcome(self, experiment: Experiment, evaluator: BaseAnswerEvaluator, throughput: Throughput) -> Preferences:
        name = str(evaluator.config.evaluator_name)
        n_games = correct = ties = 0
        usages = []
        points_by_grades: dict[tuple[int, int], list[float]] = {}
        points_by_query: dict[str, list[float]] = {}
        for i, pair in enumerate(self.pairs):
            query = experiment[f"{pair.qid}_{i}"]
            grades = self.data.qrels[pair.qid][pair.better], self.data.qrels[pair.qid][pair.worse]
            for game in query.pairwise_games.values():
                result = game.evaluations.get(name)
                if result is None or result.answer is None:
                    continue
                n_games += 1
                if usage := game_usage(result):
                    usages.append(usage)
                winner = {"A": game.agent_a_answer.agent, "B": game.agent_b_answer.agent}.get(result.answer.winner)
                is_correct = winner == (query.metadata or {})["better"]
                ties += winner is None
                correct += is_correct
                points = 0.5 if winner is None else float(is_correct)
                points_by_grades.setdefault(grades, []).append(points)
                points_by_query.setdefault(pair.qid, []).append(points)
        return Preferences(
            n_games=n_games,
            n_failed=len(self.pairs) - n_games,
            correct=correct,
            ties=ties,
            usages=usages,
            throughput=throughput,
            points_by_grades=points_by_grades,
            points_by_query=points_by_query,
        )

    def tables(
        self, outcomes: Mapping[tuple[str, str], Preferences], prices: Mapping[str, Price]
    ) -> list[RenderableType]:
        table = Table(title="Preference for the passage the assessors graded higher")
        for column in (
            "evaluator",
            "model",
            "games",
            "failed",
            "accuracy",
            "accuracy (mean over queries)",
            "ties",
            "input tokens",
            "cached tokens",
            "output tokens",
            "USD / 1k games",
            "games / s",
        ):
            table.add_column(column)
        by_grades = Table(
            title="Accuracy by the human grades of the two passages",
            caption="Most sampled pairs have a grade 0 passage, and a 1 vs 0 pair has no passage that answers "
            "the query.",
        )
        for column in ("evaluator", "model", "better vs worse", "games", "accuracy"):
            by_grades.add_column(column)
        for (name, model), found in outcomes.items():
            table.add_row(
                name,
                model,
                str(found.n_games),
                str(found.n_failed),
                "-" if found.accuracy is None else f"{found.accuracy:.3f}",
                "-" if found.accuracy_over_queries is None else f"{found.accuracy_over_queries:.3f}",
                f"{found.ties / found.n_games:.1%}" if found.n_games else "-",
                *usage_cells(found.usages, found.n_games, prices.get(model)),
                found.throughput.cell(),
            )
            for grades, points in sorted(found.points_by_grades.items()):
                by_grades.add_row(name, model, f"{grades[0]} vs {grades[1]}", str(len(points)), f"{fmean(points):.3f}")
        return [table, by_grades]
