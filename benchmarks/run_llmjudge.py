"""Judges LLMJudge pairs with every retrieval evaluator variant on every model, one cached experiment each."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.table import Table

from benchmarks import llmjudge, trec_rag24
from benchmarks.agreement import Agreement, agreement, spearman_interval
from benchmarks.jev_criteria import RDNAM_CRITERIA, TREC_CRITERIA, JevCriteriaEvaluator
from benchmarks.jev_state import JsonStateTransport
from benchmarks.llmjudge import GRADES, LLMJudgeData, Split, sample, to_experiment
from benchmarks.throughput import Run, Throughput, record
from ragelo import get_llm_provider, get_retrieval_evaluator
from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator
from ragelo.llm_providers import VercelJevProvider
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import JevRubricCoverageEvaluatorConfig
from ragelo.types.formats import LLMUsage

# The reasoner's grades until the answer-focused wording replaced them, kept to reproduce that comparison.
TOPICAL_GRADES = [
    "Not relevant: The document is not on topic.",
    "Somewhat relevant: The document is on topic but does not fully answer the user question.",
    "Very relevant: The document is on topic and answers the user question.",
]
LOWER_WHEN_UNSURE = "If you are uncertain between two relevance grades, choose the lower one."

# TypeSafe's guidance for Jev: the state holds content only, the instructions are one question, and the
# levels describe situations. These are RDNAM's report test and framing, rewritten along those lines.
JEV_RDNAM_STATE = """
    [query]
    {{ query.query }}
    {%- if query.metadata and query.metadata.description %}

    [what the searcher was looking for]
    {{ query.metadata.description }}
    {%- endif %}

    [document]
    {{ document.text }}"""
# The values go through `set` because the prompt validator only recognises placeholders without filters.
JEV_RDNAM_JSON_STATE = (
    "{%- set query_json = query.query | tojson -%}{%- set document_json = document.text | tojson -%}"
    '{"query": {{ query_json }}, "document": {{ document_json }}}'
)
JEV_REPORT_JSON_STATE = (
    "{%- set question_json = query.query | tojson -%}{%- set document_json = document.text | tojson -%}"
    '{"question": {{ question_json }}, "document": {{ document_json }}}'
)
JEV_RDNAM_QUESTION = "How useful is the document to someone who typed the query into a search engine?"
JEV_RDNAM_JSON_QUESTION = "How useful is `document` to someone who typed `query` into a search engine?"
JEV_RDNAM_LEVELS = [
    "Nothing in the document would be used in a report on the topic of the query.",
    (
        "Some information in the document would be used in a report on the topic of the query, "
        "but the document is not mainly about that topic."
    ),
    "The document is primarily about the topic of the query, or contains vital information about it.",
]

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
    "jev_boolean_report_json": {
        "evaluator_name": "jev",
        "user_prompt": JEV_REPORT_JSON_STATE,
        "system_prompt": "Assume you are writing a report on the topic of `question`. "
        "Would you use any of the information contained in `document` in that report?",
    },
    "jev_score": {"evaluator_name": "jev", "boolean_question": False},
    "jev_score_topical": {"evaluator_name": "jev", "boolean_question": False, "relevance_grades": TOPICAL_GRADES},
    "jev_score_0_3": {"evaluator_name": "jev", "boolean_question": False, "relevance_grades": GRADES},
    "jev_rdnam": {"evaluator_name": "jev_rdnam"},
    "jev_rdnam_aspects": {"evaluator_name": "jev_rdnam", "use_aspects": True},
    "jev_rdnam_state": {"evaluator_name": "jev_rdnam", "user_prompt": JEV_RDNAM_STATE},
    "jev_rdnam_question": {"evaluator_name": "jev_rdnam", "system_prompt": JEV_RDNAM_QUESTION},
    "jev_rdnam_levels": {"evaluator_name": "jev_rdnam", "relevance_grades": JEV_RDNAM_LEVELS},
    "jev_rdnam_all": {
        "evaluator_name": "jev_rdnam",
        "user_prompt": JEV_RDNAM_STATE,
        "system_prompt": JEV_RDNAM_QUESTION,
        "relevance_grades": JEV_RDNAM_LEVELS,
    },
    "jev_rdnam_json": {
        "evaluator_name": "jev_rdnam",
        "user_prompt": JEV_RDNAM_JSON_STATE,
        "system_prompt": JEV_RDNAM_JSON_QUESTION,
        "relevance_grades": JEV_RDNAM_LEVELS,
    },
    "jev_rdnam_criteria": {"evaluator_name": "jev_rubric_coverage"},
    "jev_trec_criteria": {"evaluator_name": "jev_rubric_coverage", "relevance_grades": GRADES},
}
# Variants judged against the same criteria for every query, by an evaluator that only lives in the benchmark.
JSON_STATE = {"jev_rdnam_json", "jev_boolean_report_json"}
CRITERIA = {"jev_rdnam_criteria": RDNAM_CRITERIA, "jev_trec_criteria": TREC_CRITERIA}


def build_evaluator(
    variant: str, llm_provider: BaseLLMProvider, qids: Iterable[str], n_processes: int
) -> BaseRetrievalEvaluator:
    kwargs = EVALUATORS[variant] | {"n_processes": n_processes}
    if variant in JSON_STATE and isinstance(llm_provider, VercelJevProvider):
        client = httpx.AsyncClient(
            transport=JsonStateTransport(httpx.AsyncHTTPTransport()), timeout=llm_provider.config.timeout
        )
        llm_provider = VercelJevProvider(llm_provider.config, client=client)
    if variant not in CRITERIA:
        return get_retrieval_evaluator(llm_provider=llm_provider, **kwargs)
    config = JevRubricCoverageEvaluatorConfig(**kwargs, rubrics=dict.fromkeys(qids, CRITERIA[variant]))
    return JevCriteriaEvaluator.from_config(config, llm_provider)


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
            evaluator = build_evaluator(variant, llm_provider, data.queries, n_processes)
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
