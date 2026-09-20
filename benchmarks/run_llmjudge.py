"""Judges LLMJudge pairs with every retrieval evaluator variant on every model, one cached experiment each."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from benchmarks import llmjudge, trec_rag24
from benchmarks.agreement import Agreement, agreement, spearman_interval
from benchmarks.llmjudge import GRADES, LLMJudgeData, Split, sample, to_experiment
from benchmarks.throughput import Run, Throughput, record
from ragelo import get_llm_provider, get_retrieval_evaluator
from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator
from ragelo.types.formats import LLMUsage

# The reasoner's grades until the answer-focused wording replaced them, kept to reproduce that comparison.
TOPICAL_GRADES = [
    "Not relevant: The document is not on topic.",
    "Somewhat relevant: The document is on topic but does not fully answer the user question.",
    "Very relevant: The document is on topic and answers the user question.",
]
LOWER_WHEN_UNSURE = "If you are uncertain between two relevance grades, choose the lower one."

EVALUATORS: dict[str, dict[str, Any]] = {
    "reasoner": {"evaluator_name": "reasoner"},
    "rdnam": {"evaluator_name": "RDNAM"},
    "rdnam_aspects": {"evaluator_name": "RDNAM", "use_aspects": True},
    "rdnam_annotators": {"evaluator_name": "RDNAM", "use_multiple_annotators": True},
    "domain_expert": {"evaluator_name": "domain_expert", "expert_in": "web search"},
    "reasoner_0_3": {"evaluator_name": "reasoner", "relevance_grades": GRADES},
    "rdnam_0_3": {"evaluator_name": "RDNAM", "relevance_grades": GRADES},
    "domain_expert_0_3": {"evaluator_name": "domain_expert", "expert_in": "web search", "relevance_grades": GRADES},
    "reasoner_topical": {"evaluator_name": "reasoner", "relevance_grades": TOPICAL_GRADES},
    "reasoner_strict": {"evaluator_name": "reasoner", "guidelines": LOWER_WHEN_UNSURE},
    "jev_boolean": {"evaluator_name": "jev"},
    "jev_boolean_helps": {
        "evaluator_name": "jev",
        "system_prompt": "Does the document contain information that helps answer the user question?",
    },
    "jev_score": {"evaluator_name": "jev", "boolean_question": False},
    "jev_score_topical": {"evaluator_name": "jev", "boolean_question": False, "relevance_grades": TOPICAL_GRADES},
    "jev_score_0_3": {"evaluator_name": "jev", "boolean_question": False, "relevance_grades": GRADES},
    "jev_rdnam": {"evaluator_name": "jev_rdnam"},
    "jev_rdnam_aspects": {"evaluator_name": "jev_rdnam", "use_aspects": True},
}

# Each dataset module has `download(data_dir)` and `load(data_dir, split)`, and lists its splits, default first.
DATASETS = {"llmjudge": (llmjudge, ("dev", "test")), "trec_rag24": (trec_rag24, ("test",))}

app = typer.Typer()


@dataclass(frozen=True, slots=True)
class Price:
    """USD per million tokens. Cached input tokens are billed at `cached`, the rest of the input at `input`."""

    input: float
    output: float
    cached: float

    @classmethod
    def parse(cls, spec: str) -> tuple[str, Price]:
        model, _, rates = spec.rpartition("=")
        values = [float(rate) for rate in rates.split(",")]
        if not model or len(values) != 3:
            raise typer.BadParameter(f"Expected MODEL=INPUT,OUTPUT,CACHED in USD per million tokens, got {spec}")
        return model, cls(*values)

    def cost(self, usage: LLMUsage) -> float:
        fresh_input = usage.input_tokens - usage.cached_tokens
        return (
            fresh_input * self.input + usage.cached_tokens * self.cached + usage.output_tokens * self.output
        ) / 1_000_000


# Jev bills input tokens only, though it reports a few output tokens per call. USD 0.042 per million as of
# 2026-09-19, https://vercel.com/kb/guide/typesafe-jev-and-ai-sdk. A --price for the same model replaces it.
DEFAULT_PRICES = {"typesafe-ai/jev": Price(input=0.042, output=0.0, cached=0.042)}


@dataclass(frozen=True, slots=True)
class Judgments:
    """The scores as stored, before `get_qrels` rounds them to labels, and the tokens each judgment was billed for.

    A failed evaluation is not stored, so it has no score and no usage: `n_failed` counts them, and the
    tokens a failed call was billed for are unknown. Judgments cached before usage was recorded have none
    either, so `usages` can be shorter than the scores.
    """

    scores: dict[str, dict[str, float]]
    usages: list[LLMUsage]
    n_failed: int
    throughput: Throughput


def judge(
    data: LLMJudgeData,
    evaluator: BaseRetrievalEvaluator,
    experiment_name: str,
    results_dir: Path,
    clock: Callable[[], float] = time.perf_counter,
) -> Judgments:
    """Evaluations are cached per experiment, so a re-run only judges the pairs that are missing or failed."""
    experiment = to_experiment(data, experiment_name, save_path=str(results_dir / f"{experiment_name}.json"))
    name = str(evaluator.config.evaluator_name)
    documents = [document for query in experiment for document in query.retrieved_docs.values()]
    cached = sum(name in document.evaluations for document in documents)
    started = clock()
    evaluator.evaluate_experiment(experiment)
    seconds = clock() - started
    judged = [document.evaluations[name] for document in documents if name in document.evaluations]
    scores: dict[str, dict[str, float]] = {}
    for result in judged:
        if result.score is not None:
            scores.setdefault(result.qid, {})[result.did] = result.score
    run = Run(evaluations=len(judged) - cached, seconds=seconds, n_processes=evaluator.config.n_processes)
    return Judgments(
        scores=scores,
        usages=[result.usage for result in judged if result.usage and result.score is not None],
        n_failed=len(documents) - sum(len(judged_docs) for judged_docs in scores.values()),
        throughput=record(results_dir / f"{experiment_name}_throughput.json", run),
    )


@dataclass(frozen=True, slots=True)
class Row:
    """`agreement` and `interval` are computed over the pairs every row of the table judged, so that a
    judge that failed on the hard passages is not compared on an easier subset."""

    agreement: Agreement
    interval: tuple[float, float]
    judgments: Judgments
    price: Price | None


def usage_cells(usages: list[LLMUsage], n_judged: int, price: Price | None) -> list[str]:
    """Blank unless every judged item has its usage: a mean over the few that do says nothing about the rest."""
    if len(usages) < n_judged or not usages:
        return ["-", "-", "-", "-"]
    n = len(usages)
    cells = [
        f"{sum(usage.input_tokens for usage in usages) / n:.0f}",
        f"{sum(usage.cached_tokens for usage in usages) / n:.0f}",
        f"{sum(usage.output_tokens for usage in usages) / n:.0f}",
    ]
    if price is None:
        return [*cells, "-"]
    return [*cells, f"{1000 * sum(price.cost(usage) for usage in usages) / n:.4f}"]


def common_pairs(judged: Iterable[Judgments]) -> set[tuple[str, str]]:
    """The pairs every judge scored. A judge that scored nothing gets an empty row, and is left out here
    because it would otherwise leave every other row empty too."""
    scored = [{(qid, did) for qid, scores in judgments.scores.items() for did in scores} for judgments in judged]
    return set.intersection(*(pairs for pairs in scored if pairs)) if any(scored) else set()


def rank(value: float) -> tuple[bool, float]:
    """Sort key for best first. A nan compares false with everything, which leaves a sort in any order."""
    return math.isnan(value), -value


def agreement_table(rows: dict[tuple[str, str], Row], n_common: int, n_pairs: int) -> Table:
    table = Table(
        title=f"Agreement with the human labels on the {n_common} of {n_pairs} pairs every row judged",
        caption="Ranked by the Spearman of the rounded labels, which fractional and integer judges share. "
        "Its interval resamples queries. Tokens and cost cover judged pairs only.",
    )
    for column in (
        "evaluator",
        "model",
        "scale",
        "failed",
        "kappa (binary)",
        "kappa (graded)",
        "alpha (ordinal)",
        "spearman",
        "95% interval",
        "spearman (raw score)",
        "input tokens",
        "cached tokens",
        "output tokens",
        "USD / 1k judged",
        "evaluations / s",
    ):
        table.add_column(column)
    for (variant, model), row in sorted(rows.items(), key=lambda item: rank(item[1].agreement.spearman)):
        result, judgments = row.agreement, row.judgments
        n_judged = sum(len(scores) for scores in judgments.scores.values())
        table.add_row(
            variant,
            model,
            f"0-{result.max_label}",
            str(judgments.n_failed),
            *(f"{value:.3f}" for value in (result.kappa_binary, result.kappa_graded, result.alpha_ordinal)),
            f"{result.spearman:.3f}",
            f"{row.interval[0]:.3f} to {row.interval[1]:.3f}",
            f"{result.spearman_raw:.3f}",
            *usage_cells(judgments.usages, n_judged, row.price),
            judgments.throughput.cell(),
        )
    return table


@app.command()
def main(
    models: list[str] = typer.Option(..., "--model", help="Model to judge with. Repeat for several."),
    evaluators: list[str] = typer.Option(list(EVALUATORS), "--evaluator", help="Evaluator variant. Repeat."),
    provider: str = typer.Option("openai", help="The LLM provider every model is called through."),
    dataset: str = typer.Option("llmjudge", help=f"One of {', '.join(DATASETS)}."),
    split: Split | None = typer.Option(None, help="Defaults to the dataset's first split."),
    n_pairs: int | None = typer.Option(None, help="Judge a label-stratified sample of about this many pairs."),
    seed: int = typer.Option(42, help="Seed of the sample."),
    n_processes: int = typer.Option(16, help="Parallel LLM calls."),
    prices: list[str] = typer.Option(
        [],
        "--price",
        help="MODEL=INPUT,OUTPUT,CACHED in USD per million tokens, to report the cost per 1,000 pairs. Repeat.",
    ),
    data_dir: Path | None = typer.Option(None, help="Defaults to benchmarks/data/<dataset>."),
    results_dir: Path | None = typer.Option(None, help="Defaults to benchmarks/results/<dataset>."),
    tag: str = typer.Option("", help="Suffix of the experiment names, to judge again without reusing a cache."),
) -> None:
    if dataset not in DATASETS:
        raise typer.BadParameter(f"Unknown dataset {dataset}. Choose one of {', '.join(DATASETS)}.")
    module, splits = DATASETS[dataset]
    split = split or splits[0]  # type: ignore[assignment]
    if split not in splits:
        raise typer.BadParameter(f"{dataset} has no {split} split. Choose one of {', '.join(splits)}.")
    data_dir = data_dir or Path("benchmarks/data") / dataset
    results_dir = results_dir or Path("benchmarks/results") / dataset
    module.download(data_dir)
    data = module.load(data_dir, split)
    subset = f"{split}_all"
    if n_pairs is not None:
        data = sample(data, n_pairs, seed)
        subset = f"{split}_{n_pairs}_seed{seed}"

    price_of = DEFAULT_PRICES | dict(Price.parse(spec) for spec in prices)
    judged: dict[tuple[str, str], tuple[Judgments, int]] = {}
    for model in models:
        llm_provider = get_llm_provider(provider, model=model)
        for variant in evaluators:
            evaluator = get_retrieval_evaluator(
                llm_provider=llm_provider, n_processes=n_processes, **EVALUATORS[variant]
            )
            experiment_name = f"{subset}_{variant}_{model.replace('/', '_')}{tag and '_' + tag}"
            judged[(variant, model)] = judge(data, evaluator, experiment_name, results_dir), evaluator.max_score

    common = common_pairs([judgments for judgments, _ in judged.values()])
    rows = {}
    for key, (judgments, max_score) in judged.items():
        scores = {
            qid: {did: score for did, score in by_did.items() if (qid, did) in common}
            for qid, by_did in judgments.scores.items()
        }
        rows[key] = Row(
            agreement=agreement(data.qrels, scores, max_label=max_score),
            interval=spearman_interval(data.qrels, scores),
            judgments=judgments,
            price=price_of.get(key[1]),
        )
    Console().print(agreement_table(rows, len(common), data.n_pairs))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    app()
