"""Label-level agreement between two sets of qrels over the pairs both of them judged."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Qrels = Mapping[str, Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class Agreement:
    n_pairs: int
    max_label: int
    kappa_binary: float
    kappa_graded: float
    alpha_ordinal: float
    spearman: float
    spearman_raw: float


def agreement(reference: Qrels, judged: Qrels, max_label: int = 2, relevant_from: int = 2) -> Agreement:
    """Kappa and alpha need both sides on one scale, so labels above `max_label` are collapsed into it.

    That puts the 0-3 TREC labels and the 0-2 RAGElo labels on the same three grades. Spearman is
    rank-based and reads the uncollapsed labels. Fractional scores are rounded half-up to labels, as
    `Experiment.get_qrels` does, except for `spearman_raw`, which keeps the ranking the rounding loses.
    """
    scores = [
        (int(reference[qid][did]), score)
        for qid, labels in judged.items()
        for did, score in labels.items()
        if did in reference.get(qid, {})
    ]
    pairs = [(label, math.floor(score + 0.5)) for label, score in scores]
    raw_reference = [a for a, _ in pairs]
    raw_judged = [b for _, b in pairs]
    graded_reference = [min(a, max_label) for a in raw_reference]
    graded_judged = [min(b, max_label) for b in raw_judged]
    return Agreement(
        n_pairs=len(pairs),
        max_label=max_label,
        kappa_binary=cohen_kappa(
            [int(a >= relevant_from) for a in raw_reference], [int(b >= relevant_from) for b in raw_judged]
        ),
        kappa_graded=cohen_kappa(graded_reference, graded_judged),
        alpha_ordinal=krippendorff_alpha_ordinal(graded_reference, graded_judged),
        spearman=spearman(raw_reference, raw_judged),
        spearman_raw=spearman(raw_reference, [score for _, score in scores]),
    )


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    n = len(a)
    if n == 0:
        return math.nan
    observed = sum(x == y for x, y in zip(a, b)) / n
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum(counts_a[label] * counts_b[label] for label in counts_a) / n**2
    if expected == 1:
        return math.nan
    return (observed - expected) / (1 - expected)


def krippendorff_alpha_ordinal(a: Sequence[int], b: Sequence[int]) -> float:
    """Two raters, no missing labels. Krippendorff (2011), "Computing Krippendorff's Alpha-Reliability"."""
    coincidences: Counter[tuple[int, int]] = Counter()
    for x, y in zip(a, b):
        coincidences[(x, y)] += 1
        coincidences[(y, x)] += 1
    totals: Counter[int] = Counter()
    for (x, _), count in coincidences.items():
        totals[x] += count
    n = sum(totals.values())

    def distance(c: int, k: int) -> float:
        low, high = min(c, k), max(c, k)
        return (sum(totals[g] for g in range(low, high + 1)) - (totals[low] + totals[high]) / 2) ** 2

    observed = sum(count * distance(c, k) for (c, k), count in coincidences.items())
    expected = sum(totals[c] * totals[k] * distance(c, k) for c in totals for k in totals)
    if expected == 0:
        return math.nan
    return 1 - (n - 1) * observed / expected


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ranks_a, ranks_b = _average_ranks(a), _average_ranks(b)
    n = len(a)
    if n == 0:
        return math.nan
    mean_a, mean_b = sum(ranks_a) / n, sum(ranks_b) / n
    covariance = sum((x - mean_a) * (y - mean_b) for x, y in zip(ranks_a, ranks_b))
    spread = math.sqrt(sum((x - mean_a) ** 2 for x in ranks_a) * sum((y - mean_b) ** 2 for y in ranks_b))
    if spread == 0:
        return math.nan
    return covariance / spread


def _average_ranks(values: Sequence[float]) -> list[float]:
    counts = Counter(values)
    rank_of: dict[float, float] = {}
    seen = 0
    for value in sorted(counts):
        rank_of[value] = seen + (counts[value] + 1) / 2
        seen += counts[value]
    return [rank_of[value] for value in values]
