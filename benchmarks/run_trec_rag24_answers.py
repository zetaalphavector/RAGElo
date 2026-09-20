"""System ranking on the TREC 2024 RAG answers: an Elo tournament over pairwise games, against the ranking of
the same systems by NIST's nugget scores."""

from __future__ import annotations

import logging
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from scipy.stats import kendalltau, spearmanr

from benchmarks.run_llmjudge import DEFAULT_PRICES, Price, usage_cells
from benchmarks.throughput import Run, Throughput, record
from benchmarks.trec_rag24_answers import (
    AnswerData,
    download,
    human_fulfillments,
    load,
    nugget_score,
    rubrics,
    sample_topics,
    to_experiment,
)
from ragelo import get_agent_ranker, get_answer_evaluator, get_llm_provider
from ragelo.evaluators.answer_evaluators import BaseAnswerEvaluator
from ragelo.types.formats import LLMUsage

app = typer.Typer()


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


def play(
    data: AnswerData,
    evaluator: BaseAnswerEvaluator,
    experiment_name: str,
    results_dir: Path,
    seed: int = 42,
    clock: Callable[[], float] = time.perf_counter,
) -> Tournament:
    experiment = to_experiment(data, experiment_name, save_path=str(results_dir / f"{experiment_name}.json"))
    name = str(evaluator.config.evaluator_name)
    cached = sum(name in game.evaluations for query in experiment for game in query.pairwise_games.values())
    started = clock()
    evaluator.evaluate_experiment(experiment)
    seconds = clock() - started

    n_games = decided = agreeing = ties = total = 0
    usages = []
    for query in experiment:
        human = {
            run: nugget_score(data.nuggets[query.qid], human_fulfillments(data, query.qid, run))
            for run in data.answers[query.qid]
        }
        for game in query.pairwise_games.values():
            total += 1
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
            agent_a, agent_b = game.agent_a_answer.agent, game.agent_b_answer.agent
            winner = {"A": agent_a, "B": agent_b}.get(result.answer.winner)
            ties += winner is None
            if human[agent_a] != human[agent_b] and winner is not None:
                decided += 1
                agreeing += winner == max((agent_a, agent_b), key=human.__getitem__)

    # The ranker shuffles the games of each of its tournaments with the global generator.
    random.seed(seed)
    ranking = get_agent_ranker("elo").run(experiment)
    run = Run(n_games - cached, seconds, evaluator.config.n_processes, calls_per_evaluation=2)
    return Tournament(
        elo=ranking.scores,
        # The ranker adds up the wins of every tournament it replays the same games in.
        wins={run: wins // ranking.total_tournaments for run, wins in ranking.wins.items()},
        n_games=n_games,
        n_failed=total - n_games,
        decided=decided,
        agreeing=agreeing,
        ties=ties,
        usages=usages,
        throughput=record(results_dir / f"{experiment_name}_throughput.json", run),
    )


@app.command()
def main(
    evaluators: list[str] = typer.Option(["jev_pairwise"], "--evaluator", help="Pairwise answer evaluator. Repeat."),
    model: str = typer.Option("typesafe-ai/jev"),
    provider: str = typer.Option("vercel-jev"),
    n_systems: int = typer.Option(10, help="The best, the worst and a random draw of the systems between them."),
    n_topics: int | None = typer.Option(None, help="A difficulty-stratified sample of topics. All 56 by default."),
    seed: int = typer.Option(42, help="Seed of the system draw, the topic sample and the Elo tournaments."),
    n_processes: int = typer.Option(16, help="Parallel games. Each game is judged in both answer orders."),
    data_dir: Path = typer.Option(Path("benchmarks/data/trec_rag24_answers")),
    results_dir: Path = typer.Option(Path("benchmarks/results/trec_rag24_answers")),
    tag: str = typer.Option("", help="Suffix of the experiment names, to judge again without reusing a cache."),
    prices: list[str] = typer.Option([], "--price", help="MODEL=INPUT,OUTPUT,CACHED in USD per million tokens."),
) -> None:
    download(data_dir)
    everything = load(data_dir)
    systems = pick_systems(everything, n_systems, seed)
    data = keep_systems(sample_topics(everything, n_topics, seed) if n_topics else everything, systems)
    human = human_scores(data)
    price = (DEFAULT_PRICES | dict(Price.parse(spec) for spec in prices)).get(model)
    results_dir.mkdir(parents=True, exist_ok=True)
    llm_provider = get_llm_provider(provider, model=model)
    console = Console()

    for name in evaluators:
        options: dict[str, Any] = {"rubrics": rubrics(data)} if "rubric" in name else {}
        evaluator = get_answer_evaluator(name, llm_provider=llm_provider, n_processes=n_processes, **options)
        subset = f"{len(data.queries)}topics_{n_systems}systems_seed{seed}"
        played = play(
            data, evaluator, f"{subset}_{name}_{model.replace('/', '_')}{tag and '_' + tag}", results_dir, seed
        )

        table = Table(title=f"{name} on {model}: {len(systems)} systems, {len(data.queries)} topics")
        for column in ("system", "human vital strict", "human rank", "Elo", "Elo rank", "games won"):
            table.add_column(column)
        by_human = sorted(systems, key=lambda run: -human[run])
        by_elo = sorted(systems, key=lambda run: -played.elo.get(run, 0.0))
        for run in by_human:
            table.add_row(
                run,
                f"{human[run]:.3f}",
                str(by_human.index(run) + 1),
                f"{played.elo.get(run, float('nan')):.0f}",
                str(by_elo.index(run) + 1),
                str(played.wins.get(run, 0)),
            )
        console.print(table)

        ratings = [played.elo.get(run, 0.0) for run in by_human]
        scores = [human[run] for run in by_human]
        tokens = usage_cells(played.usages, played.n_games, price)
        agreement = f"{played.agreeing / played.decided:.1%}" if played.decided else "-"
        console.print(
            f"Kendall tau {kendalltau(scores, ratings).statistic:.3f} and Spearman "
            f"{spearmanr(scores, ratings).statistic:.3f} between the Elo ratings and the human scores.\n"
            f"Games: {played.n_games} judged, {played.n_failed} failed, {played.ties} tied. The winner is the answer "
            f"with the higher human score in {played.agreeing} of the {played.decided} games that have one ({agreement}).\n"
            f"Per game: {tokens[0]} input and {tokens[2]} output tokens, {tokens[3]} USD per 1,000 games, "
            f"{played.throughput.cell()} games per second.\n"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    app()
