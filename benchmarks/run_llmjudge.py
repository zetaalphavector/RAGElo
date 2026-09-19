"""Judges LLMJudge pairs with every retrieval evaluator variant on every model, one cached experiment each."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from benchmarks.agreement import Agreement, agreement
from benchmarks.llmjudge import GRADES, LLMJudgeData, Split, download, load, sample, to_experiment
from ragelo import get_llm_provider, get_retrieval_evaluator
from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator

EVALUATORS: dict[str, dict[str, Any]] = {
    "reasoner": {"evaluator_name": "reasoner"},
    "rdnam": {"evaluator_name": "RDNAM"},
    "rdnam_aspects": {"evaluator_name": "RDNAM", "use_aspects": True},
    "rdnam_annotators": {"evaluator_name": "RDNAM", "use_multiple_annotators": True},
    "domain_expert": {"evaluator_name": "domain_expert", "expert_in": "web search"},
    "reasoner_0_3": {"evaluator_name": "reasoner", "relevance_grades": GRADES},
    "rdnam_0_3": {"evaluator_name": "RDNAM", "relevance_grades": GRADES},
    "domain_expert_0_3": {"evaluator_name": "domain_expert", "expert_in": "web search", "relevance_grades": GRADES},
}

app = typer.Typer()


def judge(
    data: LLMJudgeData, evaluator: BaseRetrievalEvaluator, experiment_name: str, results_dir: Path
) -> dict[str, dict[str, float]]:
    """Evaluations are cached per experiment, so a re-run only judges the pairs that are missing or failed."""
    experiment = to_experiment(data, experiment_name, save_path=str(results_dir / f"{experiment_name}.json"))
    evaluator.evaluate_experiment(experiment)
    return experiment.get_qrels(retrieval_evaluator_name=str(evaluator.config.evaluator_name))


def agreement_table(rows: dict[tuple[str, str], Agreement]) -> Table:
    table = Table(title="Agreement with the human labels")
    for column in (
        "evaluator",
        "model",
        "scale",
        "pairs",
        "kappa (binary)",
        "kappa (graded)",
        "alpha (ordinal)",
        "spearman",
    ):
        table.add_column(column)
    for (variant, model), result in sorted(rows.items(), key=lambda row: -row[1].alpha_ordinal):
        table.add_row(
            variant,
            model,
            f"0-{result.max_label}",
            str(result.n_pairs),
            *(
                f"{value:.3f}"
                for value in (result.kappa_binary, result.kappa_graded, result.alpha_ordinal, result.spearman)
            ),
        )
    return table


@app.command()
def main(
    models: list[str] = typer.Option(..., "--model", help="Model to judge with. Repeat for several."),
    evaluators: list[str] = typer.Option(list(EVALUATORS), "--evaluator", help="Evaluator variant. Repeat."),
    provider: str = typer.Option("openai", help="The LLM provider every model is called through."),
    split: Split = typer.Option("dev"),
    n_pairs: int | None = typer.Option(None, help="Judge a label-stratified sample of about this many pairs."),
    seed: int = typer.Option(42, help="Seed of the sample."),
    n_processes: int = typer.Option(16, help="Parallel LLM calls."),
    data_dir: Path = typer.Option(Path("benchmarks/data/llmjudge")),
    results_dir: Path = typer.Option(Path("benchmarks/results/llmjudge")),
) -> None:
    download(data_dir)
    data = load(data_dir, split)
    subset = f"{split}_all"
    if n_pairs is not None:
        data = sample(data, n_pairs, seed)
        subset = f"{split}_{n_pairs}_seed{seed}"

    rows: dict[tuple[str, str], Agreement] = {}
    for model in models:
        llm_provider = get_llm_provider(provider, model=model)
        for variant in evaluators:
            evaluator = get_retrieval_evaluator(
                llm_provider=llm_provider, n_processes=n_processes, **EVALUATORS[variant]
            )
            experiment_name = f"{subset}_{variant}_{model.replace('/', '_')}"
            qrels = judge(data, evaluator, experiment_name, results_dir)
            rows[(variant, model)] = agreement(data.qrels, qrels, max_label=evaluator.max_score)
    Console().print(agreement_table(rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    app()
