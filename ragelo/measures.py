"""Resolution and classification of `ir_measures` measures.

`ir_measures` is an optional dependency (`pip install 'ragelo[eval]'`), so it is imported behind
a flag here and every helper raises a helpful ImportError when it is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ir_measures import Qrel, ScoredDoc

try:
    import ir_measures
    from ir_measures.measures import Measure

    _IR_MEASURES_AVAILABLE = True
except ImportError:
    _IR_MEASURES_AVAILABLE = False

DIVERSITY_MODULE = "ir_measures.measures.diversity"

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


def make_qrel(query_id: str, doc_id: str, relevance: int, subtopic: str | None = None) -> "Qrel":
    """Build one qrel row, optionally scoped to a subtopic.

    `ir_measures` carries the subtopic id in the `iteration` field, which is the only qrels shape
    that coverage measures can read.
    """
    if not _IR_MEASURES_AVAILABLE:
        raise ImportError(_IMPORT_ERROR)
    if subtopic is None:
        return ir_measures.Qrel(query_id, doc_id, relevance)
    return ir_measures.Qrel(query_id, doc_id, relevance, iteration=subtopic)


def make_run(run: dict[str, dict[str, float]]) -> list["ScoredDoc"]:
    if not _IR_MEASURES_AVAILABLE:
        raise ImportError(_IMPORT_ERROR)
    return [ir_measures.ScoredDoc(qid, did, score) for qid, docs in run.items() for did, score in docs.items()]


def is_coverage_measure(measure: Measure) -> bool:
    """Whether a measure scores information coverage over subtopics rather than document relevance.

    Coverage measures (`alpha_nDCG`, `StRecall`, `NRBP`, ...) reward a ranking that collectively
    addresses every subtopic of a query, so they read subtopic qrels rather than flat ones.
    """
    return type(measure).__module__ == DIVERSITY_MODULE
