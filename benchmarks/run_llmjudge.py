"""Judges LLMJudge pairs with every retrieval evaluator variant on every model, one cached experiment each."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from benchmarks.agreement import Agreement, agreement
from benchmarks.llmjudge import GRADES, LLMJudgeData, Split, download, load, sample, to_experiment
from ragelo import get_llm_provider, get_retrieval_evaluator
from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator
from ragelo.types.formats import LLMUsage

ANSWER_GRADES = [
    (
        "Not relevant: The document contains no information that helps answer the user question, "
        "even if it is on the same topic or shares keywords with it."
    ),
    "Somewhat relevant: The document contains partial or indirect information that helps answer the user question.",
    "Very relevant: The document contains the answer to the user question, even if surrounded by other content.",
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
    "reasoner_answer": {"evaluator_name": "reasoner", "relevance_grades": ANSWER_GRADES},
    "reasoner_answer_strict": {
        "evaluator_name": "reasoner",
        "relevance_grades": ANSWER_GRADES,
        "guidelines": LOWER_WHEN_UNSURE,
    },
    "jev_boolean": {"evaluator_name": "jev"},
    "jev_score": {"evaluator_name": "jev", "boolean_question": False},
    "jev_score_answer": {"evaluator_name": "jev", "boolean_question": False, "relevance_grades": ANSWER_GRADES},
    "jev_score_0_3": {"evaluator_name": "jev", "boolean_question": False, "relevance_grades": GRADES},
}

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


@dataclass(frozen=True, slots=True)
class Judgments:
    """The scores as stored, before `get_qrels` rounds them to labels, and the tokens each judgment was billed for.

    Judgments cached before usage was recorded have none, so `usages` can be shorter than the scores.
    """

    scores: dict[str, dict[str, float]]
    usages: list[LLMUsage]


def judge(data: LLMJudgeData, evaluator: BaseRetrievalEvaluator, experiment_name: str, results_dir: Path) -> Judgments:
    """Evaluations are cached per experiment, so a re-run only judges the pairs that are missing or failed."""
    experiment = to_experiment(data, experiment_name, save_path=str(results_dir / f"{experiment_name}.json"))
    evaluator.evaluate_experiment(experiment)
    name = str(evaluator.config.evaluator_name)
    scores: dict[str, dict[str, float]] = {}
    usages = []
    for query in experiment:
        for did, document in query.retrieved_docs.items():
            result = document.evaluations.get(name)
            if result is None or result.score is None:
                continue
            scores.setdefault(query.qid, {})[did] = result.score
            if result.usage:
                usages.append(result.usage)
    return Judgments(scores=scores, usages=usages)


@dataclass(frozen=True, slots=True)
class Row:
    agreement: Agreement
    usages: list[LLMUsage]
    price: Price | None

    def usage_cells(self) -> list[str]:
        if not self.usages:
            return ["-", "-", "-", "-"]
        n = len(self.usages)
        cells = [
            f"{sum(usage.input_tokens for usage in self.usages) / n:.0f}",
            f"{sum(usage.cached_tokens for usage in self.usages) / n:.0f}",
            f"{sum(usage.output_tokens for usage in self.usages) / n:.0f}",
        ]
        if self.price is None:
            return [*cells, "-"]
        return [*cells, f"{1000 * sum(self.price.cost(usage) for usage in self.usages) / n:.4f}"]


def agreement_table(rows: dict[tuple[str, str], Row]) -> Table:
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
        "spearman (raw score)",
        "input tokens",
        "cached tokens",
        "output tokens",
        "USD / 1k pairs",
    ):
        table.add_column(column)
    for (variant, model), row in sorted(rows.items(), key=lambda item: -item[1].agreement.spearman_raw):
        result = row.agreement
        table.add_row(
            variant,
            model,
            f"0-{result.max_label}",
            str(result.n_pairs),
            *(
                f"{value:.3f}"
                for value in (
                    result.kappa_binary,
                    result.kappa_graded,
                    result.alpha_ordinal,
                    result.spearman,
                    result.spearman_raw,
                )
            ),
            *row.usage_cells(),
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
    prices: list[str] = typer.Option(
        [],
        "--price",
        help="MODEL=INPUT,OUTPUT,CACHED in USD per million tokens, to report the cost per 1,000 pairs. Repeat.",
    ),
    data_dir: Path = typer.Option(Path("benchmarks/data/llmjudge")),
    results_dir: Path = typer.Option(Path("benchmarks/results/llmjudge")),
) -> None:
    download(data_dir)
    data = load(data_dir, split)
    subset = f"{split}_all"
    if n_pairs is not None:
        data = sample(data, n_pairs, seed)
        subset = f"{split}_{n_pairs}_seed{seed}"

    price_of = dict(Price.parse(spec) for spec in prices)
    rows: dict[tuple[str, str], Row] = {}
    for model in models:
        llm_provider = get_llm_provider(provider, model=model)
        for variant in evaluators:
            evaluator = get_retrieval_evaluator(
                llm_provider=llm_provider, n_processes=n_processes, **EVALUATORS[variant]
            )
            experiment_name = f"{subset}_{variant}_{model.replace('/', '_')}"
            judgments = judge(data, evaluator, experiment_name, results_dir)
            rows[(variant, model)] = Row(
                agreement=agreement(data.qrels, judgments.scores, max_label=evaluator.max_score),
                usages=judgments.usages,
                price=price_of.get(model),
            )
    Console().print(agreement_table(rows))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    app()
