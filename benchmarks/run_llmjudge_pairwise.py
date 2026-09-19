"""Pairwise preference on LLMJudge: two passages of one query with different human labels stand in for two
agents' answers, and the judge should prefer the one the assessors graded higher."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import typer
from rich.console import Console
from rich.table import Table

from benchmarks.llmjudge import LLMJudgeData, Split, download, load
from benchmarks.run_llmjudge import DEFAULT_PRICES, Price, usage_cells
from benchmarks.throughput import Run, Throughput, record
from ragelo import Experiment, get_answer_evaluator, get_llm_provider
from ragelo.evaluators.answer_evaluators import BaseAnswerEvaluator
from ragelo.types.formats import LLMUsage

MAX_PASSAGE_CHARS = 5_000
app = typer.Typer()


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


def sample_pairs(data: LLMJudgeData, n_pairs: int, seed: int = 42) -> list[PreferencePair]:
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


def to_experiment(data: LLMJudgeData, pairs: list[PreferencePair], name: str, save_path: str) -> Experiment:
    """One query per pair. The better passage is agent `a` or `b` at random, so its name gives nothing away."""
    rng = random.Random(name)
    experiment = Experiment(name, save_path=save_path)
    for i, pair in enumerate(pairs):
        qid = f"{pair.qid}_{i}"
        better_agent, worse_agent = rng.sample(["a", "b"], 2)
        experiment.add_query(
            data.queries[pair.qid], query_id=qid, metadata={"better": better_agent}, should_save=False
        )
        experiment.add_agent_answer(data.passages[pair.better], better_agent, qid, exist_ok=True, should_save=False)
        experiment.add_agent_answer(data.passages[pair.worse], worse_agent, qid, exist_ok=True, should_save=False)
    experiment.save()
    return experiment


def judge(
    data: LLMJudgeData,
    pairs: list[PreferencePair],
    evaluator: BaseAnswerEvaluator,
    experiment_name: str,
    results_dir: Path,
    clock: Callable[[], float] = time.perf_counter,
) -> Preferences:
    experiment = to_experiment(data, pairs, experiment_name, str(results_dir / f"{experiment_name}.json"))
    name = str(evaluator.config.evaluator_name)
    cached = sum(name in game.evaluations for query in experiment for game in query.pairwise_games.values())
    started = clock()
    evaluator.evaluate_experiment(experiment)
    seconds = clock() - started
    n_games = correct = ties = 0
    usages = []
    points_by_grades: dict[tuple[int, int], list[float]] = {}
    points_by_query: dict[str, list[float]] = {}
    for i, pair in enumerate(pairs):
        query = experiment[f"{pair.qid}_{i}"]
        grades = int(data.qrels[pair.qid][pair.better]), int(data.qrels[pair.qid][pair.worse])
        for game in query.pairwise_games.values():
            result = game.evaluations.get(name)
            if result is None or result.answer is None:
                continue
            n_games += 1
            orders = [order.usage for order in (result.a_vs_b_result, result.b_vs_a_result) if order and order.usage]
            if len(orders) == 2:
                usages.append(
                    LLMUsage(
                        input_tokens=sum(usage.input_tokens for usage in orders),
                        output_tokens=sum(usage.output_tokens for usage in orders),
                        cached_tokens=sum(usage.cached_tokens for usage in orders),
                    )
                )
            winner = {"A": game.agent_a_answer.agent, "B": game.agent_b_answer.agent}.get(result.answer.winner)
            is_correct = winner == (query.metadata or {})["better"]
            ties += winner is None
            correct += is_correct
            points = 0.5 if winner is None else float(is_correct)
            points_by_grades.setdefault(grades, []).append(points)
            points_by_query.setdefault(pair.qid, []).append(points)
    run = Run(n_games - cached, seconds, evaluator.config.n_processes, calls_per_evaluation=2)
    return Preferences(
        n_games=n_games,
        n_failed=len(pairs) - n_games,
        correct=correct,
        ties=ties,
        usages=usages,
        throughput=record(results_dir / f"{experiment_name}_throughput.json", run),
        points_by_grades=points_by_grades,
        points_by_query=points_by_query,
    )


@app.command()
def main(
    models: list[str] = typer.Option(..., "--model", help="Model to judge with. Repeat for several."),
    evaluators: list[str] = typer.Option(["jev_pairwise"], "--evaluator", help="Pairwise answer evaluator. Repeat."),
    provider: str = typer.Option("vercel-jev", help="The LLM provider every model is called through."),
    split: Split = typer.Option("dev"),
    n_pairs: int = typer.Option(500, help="How many passage pairs to judge."),
    seed: int = typer.Option(42, help="Seed of the sample."),
    n_processes: int = typer.Option(16, help="Parallel games. Each game is judged in both answer orders."),
    data_dir: Path = typer.Option(Path("benchmarks/data/llmjudge")),
    results_dir: Path = typer.Option(Path("benchmarks/results/llmjudge_pairwise")),
    tag: str = typer.Option("", help="Suffix of the experiment names, to judge again without reusing a cache."),
    prices: list[str] = typer.Option(
        [], "--price", help="MODEL=INPUT,OUTPUT,CACHED in USD per million tokens. Tokens cover both answer orders."
    ),
) -> None:
    price_of = DEFAULT_PRICES | dict(Price.parse(spec) for spec in prices)
    download(data_dir)
    data = load(data_dir, split)
    pairs = sample_pairs(data, n_pairs, seed)
    results_dir.mkdir(parents=True, exist_ok=True)

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
        caption="Most sampled pairs have a grade 0 passage, and a 1 vs 0 pair has no passage that answers the query.",
    )
    for column in ("evaluator", "model", "better vs worse", "games", "accuracy"):
        by_grades.add_column(column)
    for model in models:
        llm_provider = get_llm_provider(provider, model=model)
        for name in evaluators:
            evaluator = get_answer_evaluator(name, llm_provider=llm_provider, n_processes=n_processes)
            experiment_name = f"{split}_{n_pairs}_seed{seed}_{name}_{model.replace('/', '_')}{tag and '_' + tag}"
            found = judge(data, pairs, evaluator, experiment_name, results_dir)
            table.add_row(
                name,
                model,
                str(found.n_games),
                str(found.n_failed),
                "-" if found.accuracy is None else f"{found.accuracy:.3f}",
                "-" if found.accuracy_over_queries is None else f"{found.accuracy_over_queries:.3f}",
                f"{found.ties / found.n_games:.1%}" if found.n_games else "-",
                *usage_cells(found.usages, found.n_games, price_of.get(model)),
                found.throughput.cell(),
            )
            for grades, points in sorted(found.points_by_grades.items()):
                by_grades.add_row(name, model, f"{grades[0]} vs {grades[1]}", str(len(points)), f"{fmean(points):.3f}")
    Console().print(table)
    Console().print(by_grades)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    app()
