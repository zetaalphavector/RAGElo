"""Label-level agreement between two sets of qrels over the pairs both of them judged."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

try:
    from scipy.stats import kendalltau, spearmanr

    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

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

    `kappa_binary` cuts the rounded labels at `relevant_from`, so a fractional judge is relevant from
    `relevant_from - 0.5`. For a yes/no judge scaled to 0-2 that is a yes-probability of 0.75, not 0.5.
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


def spearman_interval(
    reference: Qrels, judged: Qrels, n_resamples: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Bootstrap interval of the label Spearman, resampling queries.

    Pairs of one query are not independent, so resampling pairs gives an interval that is too narrow.
    """
    labels = {
        qid: [
            (int(reference[qid][did]), math.floor(score + 0.5))
            for did, score in scores.items()
            if did in reference[qid]
        ]
        for qid, scores in judged.items()
        if qid in reference
    }
    qids = sorted(qid for qid, pairs in labels.items() if pairs)
    if not qids:
        return math.nan, math.nan
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_resamples):
        pairs = [pair for qid in rng.choices(qids, k=len(qids)) for pair in labels[qid]]
        estimate = spearman([a for a, _ in pairs], [b for _, b in pairs])
        if not math.isnan(estimate):
            estimates.append(estimate)
    if not estimates:
        return math.nan, math.nan
    estimates.sort()
    tail = (1 - level) / 2
    return estimates[int(tail * len(estimates))], estimates[min(int((1 - tail) * len(estimates)), len(estimates) - 1)]


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """nan for a constant side, which scipy also returns, but with a warning per call."""
    if len(set(a)) < 2 or len(set(b)) < 2:
        return math.nan
    return float(spearmanr(a, b).statistic)


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    return float(kendalltau(a, b).statistic)


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
