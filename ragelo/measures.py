"""Resolution and classification of `ir_measures` measures.

`ir_measures` is an optional dependency (`pip install 'ragelo[eval]'`), so it is imported behind
a flag here and every helper raises a helpful ImportError when it is missing.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ir_measures import Qrel, ScoredDoc

try:
    import ir_measures
    from ir_measures.measures import Measure

    _IR_MEASURES_AVAILABLE = True
except ImportError:
    _IR_MEASURES_AVAILABLE = False

_IMPORT_ERROR = (
    "ir_measures is not installed. Please install it with `pip install 'ragelo[eval]'`. "
    "Coverage measures additionally need `pip install 'ir-measures[pyndeval]'`."
)


def parse_measure(metric: str | Measure) -> Measure:
    """Resolve a measure from a `NAME`, `NAME@CUTOFF` or `NAME(param=value,...)@CUTOFF` string.

    `ir_measures.parse_measure` evaluates the parameter list through `ast.Num`, which was removed
    in Python 3.14, so every metric string fails there. Resolving against the registry directly
    keeps measure strings working on every supported interpreter. Measure objects pass through, so
    callers can build them with `ir_measures.nDCG @ 10` and skip parsing entirely.
    """
    if not _IR_MEASURES_AVAILABLE:
        raise ImportError(_IMPORT_ERROR)
    if not isinstance(metric, str):
        return metric

    body, _, cutoff = metric.rpartition("@")
    if not body:
        body, cutoff = cutoff, ""
    name, _, params = body.partition("(")
    name = name.strip()

    registry = ir_measures.measures.registry
    if name not in registry:
        raise ValueError(f"Metric {metric} not found. Valid metrics are: {sorted(registry)}")
    measure = registry[name]
    if params:
        measure = measure(**_parse_params(params.rstrip(") ")))
    if cutoff:
        measure = measure @ _parse_value(cutoff)
    return measure


def _parse_params(params: str) -> dict[str, Any]:
    parsed = {}
    for pair in params.split(","):
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"Measure parameter {pair!r} is not of the form key=value")
        parsed[key.strip()] = _parse_value(value)
    return parsed


def _parse_value(value: str) -> Any:
    value = value.strip().strip("'\"")
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def make_run(run: dict[str, dict[str, float]]) -> list[ScoredDoc]:
    if not _IR_MEASURES_AVAILABLE:
        raise ImportError(_IMPORT_ERROR)
    return [ir_measures.ScoredDoc(qid, did, score) for qid, docs in run.items() for did, score in docs.items()]


def calc_per_query(
    measures: list[Measure],
    qrels: dict[str, dict[str, float]] | list[Qrel],
    run: dict[str, dict[str, float]] | list[ScoredDoc],
) -> dict[str, dict[str, float]]:
    """Per-query scores as `{metric: {query_id: value}}`, covering only the queries in the run."""
    if not _IR_MEASURES_AVAILABLE:
        raise ImportError(_IMPORT_ERROR)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for metric in ir_measures.iter_calc(measures, qrels, run):
        scores[str(metric.measure)][metric.query_id] = metric.value
    return dict(scores)


def paired_permutation_pvalue(deltas: Sequence[float], n_permutations: int = 10_000, seed: int = 42) -> float:
    """Two-sided sign-flip permutation test over paired per-query score differences."""
    values = np.asarray(deltas, dtype=float)
    if values.size == 0 or not values.any():
        return 1.0
    observed = abs(values.mean())
    signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(n_permutations, values.size))
    permuted = np.abs((signs * values).mean(axis=1))
    return float((1 + (permuted >= observed).sum()) / (n_permutations + 1))
