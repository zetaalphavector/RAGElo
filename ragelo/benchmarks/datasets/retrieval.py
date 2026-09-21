from __future__ import annotations

import math
import random
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

from rich.console import RenderableType
from rich.table import Table

from ragelo import Experiment, get_retrieval_evaluator
from ragelo.benchmarks.agreement import agreement, spearman_interval
from ragelo.benchmarks.datasets.base import Dataset
from ragelo.benchmarks.pricing import Price, usage_cells
from ragelo.benchmarks.throughput import Throughput
from ragelo.benchmarks.variants import RETRIEVAL_VARIANTS
from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.evaluables import Document
from ragelo.types.formats import LLMUsage


@dataclass(frozen=True, slots=True)
class RetrievalData:
    queries: dict[str, str]
    passages: dict[str, str]
    qrels: dict[str, dict[str, int]]

    @property
    def n_pairs(self) -> int:
        return sum(len(judged) for judged in self.qrels.values())


@dataclass(frozen=True, slots=True)
class Judgments:
    """The scores as stored, before `get_qrels` rounds them to labels, and the tokens each judgment was billed for.

    A failed evaluation is not stored, so it has no score and no usage: `n_failed` counts them, and the
    tokens a failed call was billed for are unknown. Judgments cached before usage was recorded have none
    either, so `usages` can be shorter than the scores.
    """

    scores: dict[str, dict[str, float]]
    max_score: int
    usages: list[LLMUsage]
    n_failed: int
    throughput: Throughput


def sample(data: RetrievalData, n_pairs: int, seed: int = 42) -> RetrievalData:
    """Draws about `n_pairs` pairs, keeping the label distribution."""
    pairs_by_label: dict[int, list[tuple[str, str]]] = {}
    for qid, labels in data.qrels.items():
        for did, label in labels.items():
            pairs_by_label.setdefault(label, []).append((qid, did))

    rng = random.Random(seed)
    qrels: dict[str, dict[str, int]] = {}
    for label, pairs in sorted(pairs_by_label.items()):
        for qid, did in rng.sample(pairs, round(n_pairs * len(pairs) / data.n_pairs)):
            qrels.setdefault(qid, {})[did] = label

    judged = {did for labels in qrels.values() for did in labels}
    return RetrievalData(
        queries={qid: data.queries[qid] for qid in qrels},
        passages={did: data.passages[did] for did in judged},
        qrels=qrels,
    )


def common_pairs(judged: Iterable[Judgments]) -> set[tuple[str, str]]:
    """The pairs every judge scored. A judge that scored nothing gets an empty row, and is left out here
    because it would otherwise leave every other row empty too."""
    scored = [{(qid, did) for qid, scores in judgments.scores.items() for did in scores} for judgments in judged]
    return set.intersection(*(pairs for pairs in scored if pairs)) if any(scored) else set()


def rank(value: float) -> tuple[bool, float]:
    """Sort key for best first. A nan compares false with everything, which leaves a sort in any order."""
    return math.isnan(value), -value


class RetrievalDataset(Dataset[BaseRetrievalEvaluator, Judgments]):
    """Judged query and passage pairs, for the retrieval evaluators of `RETRIEVAL_VARIANTS`."""

    @abstractmethod
    def load(self) -> RetrievalData: ...

    @cached_property
    def data(self) -> RetrievalData:
        data = self.load()
        return data if self.config.n_samples is None else sample(data, self.config.n_samples, self.config.seed)

    @property
    def subset(self) -> str:
        return (
            f"{self.split}_all"
            if self.config.n_samples is None
            else f"{self.split}_{self.config.n_samples}_seed{self.config.seed}"
        )

    def get_evaluator(self, variant: str, llm_provider: BaseLLMProvider, **kwargs: Any) -> BaseRetrievalEvaluator:
        if variant not in RETRIEVAL_VARIANTS:
            raise ValueError(f"Unknown retrieval variant {variant}. Valid options are {list(RETRIEVAL_VARIANTS)}")
        return get_retrieval_evaluator(llm_provider=llm_provider, **RETRIEVAL_VARIANTS[variant], **kwargs)

    def to_experiment(self, experiment_name: str, save_path: str | None = None) -> Experiment:
        experiment = Experiment(experiment_name, save_path=save_path)
        for qid, text in self.data.queries.items():
            experiment.add_query(text, query_id=qid, should_save=False)
        for qid, labels in self.data.qrels.items():
            for did in labels:
                experiment.add_retrieved_doc(
                    self.data.passages[did], query_id=qid, doc_id=did, exist_ok=True, should_save=False
                )
        experiment.save()
        return experiment

    def items(self, experiment: Experiment) -> list[Document]:
        return [document for query in experiment for document in query.retrieved_docs.values()]

    def outcome(self, experiment: Experiment, evaluator: BaseRetrievalEvaluator, throughput: Throughput) -> Judgments:
        name = str(evaluator.config.evaluator_name)
        documents = self.items(experiment)
        results = [document.evaluations[name] for document in documents if name in document.evaluations]
        scores: dict[str, dict[str, float]] = {}
        for result in results:
            if result.score is not None:
                scores.setdefault(result.qid, {})[result.did] = result.score
        return Judgments(
            scores=scores,
            max_score=evaluator.max_score,
            usages=[result.usage for result in results if result.usage and result.score is not None],
            n_failed=len(documents) - sum(len(by_did) for by_did in scores.values()),
            throughput=throughput,
        )

    def tables(
        self, outcomes: Mapping[tuple[str, str], Judgments], prices: Mapping[str, Price]
    ) -> list[RenderableType]:
        """Agreement is computed over the pairs every row judged, so that a judge that failed on the hard
        passages is not compared on an easier subset."""
        common = common_pairs(outcomes.values())
        table = Table(
            title=f"Agreement with the human labels on the {len(common)} of {self.data.n_pairs} pairs every row judged",
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
        rows = []
        for (variant, model), judgments in outcomes.items():
            scores = {
                qid: {did: score for did, score in by_did.items() if (qid, did) in common}
                for qid, by_did in judgments.scores.items()
            }
            result = agreement(self.data.qrels, scores, max_label=judgments.max_score)
            low, high = spearman_interval(self.data.qrels, scores)
            n_judged = sum(len(by_did) for by_did in judgments.scores.values())
            cells = [
                variant,
                model,
                f"0-{result.max_label}",
                str(judgments.n_failed),
                *(f"{value:.3f}" for value in (result.kappa_binary, result.kappa_graded, result.alpha_ordinal)),
                f"{result.spearman:.3f}",
                f"{low:.3f} to {high:.3f}",
                f"{result.spearman_raw:.3f}",
                *usage_cells(judgments.usages, n_judged, prices.get(model)),
                judgments.throughput.cell(),
            ]
            rows.append((rank(result.spearman), cells))
        for _, cells in sorted(rows, key=lambda row: row[0]):
            table.add_row(*cells)
        return [table]
