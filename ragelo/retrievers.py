"""The contract between an Experiment and the retrieval systems it compares."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from ragelo.types.query import Query

logger = logging.getLogger(__name__)


class RetrievedDocument(BaseModel):
    """A document returned by a retriever.

    Pooling shares judgements between retrievers by `did`, so an ID must identify the same
    text everywhere. A None score is recorded as 1/(rank+1), preserving the ranking.
    """

    did: str
    text: str
    score: float | None = None
    metadata: dict[str, Any] | None = None


class Retriever(Protocol):
    """What `Experiment.run_retrievers` needs from a retrieval system."""

    async def retrieve(self, query: Query, top_k: int) -> Sequence[RetrievedDocument]:
        """Returns the top_k documents for the query, best first."""
        ...


class RunFile(BaseModel):
    """One system's ranked lists with text inline, keyed by qid. Array order is the rank."""

    retriever_name: str | None = None
    top_k: int | None = None
    runs: dict[str, list[RetrievedDocument]]


class FileRetriever:
    """Serves a pre-extracted run file as a Retriever.

    Documents without text cannot be judged, so they are dropped at load and counted in
    `skipped_without_text`.
    """

    def __init__(self, runs: dict[str, list[RetrievedDocument]], skipped_without_text: int = 0) -> None:
        self.__runs = runs
        self.skipped_without_text = skipped_without_text

    @classmethod
    def from_run_file(cls, path: str | Path) -> FileRetriever:
        run_file = RunFile.model_validate_json(Path(path).read_text())
        runs: dict[str, list[RetrievedDocument]] = {}
        skipped = 0
        for qid, documents in run_file.runs.items():
            kept = [document for document in documents if document.text]
            skipped += len(documents) - len(kept)
            runs[qid] = kept
        if skipped:
            logger.warning(f"{path}: dropped {skipped} documents without text, which cannot be judged.")
        return cls(runs, skipped)

    async def retrieve(self, query: Query, top_k: int) -> list[RetrievedDocument]:
        if query.qid not in self.__runs:
            logger.warning(f"Run file has no entry for query {query.qid}. Do the qid schemes match?")
            return []
        return self.__runs[query.qid][:top_k]


class NamespacedRetriever:
    """Prefixes every did a retriever returns, so systems searching different corpora can pool."""

    def __init__(self, retriever: Retriever, prefix: str) -> None:
        self.__retriever = retriever
        self.__prefix = prefix

    async def retrieve(self, query: Query, top_k: int) -> list[RetrievedDocument]:
        documents = await self.__retriever.retrieve(query, top_k)
        return [document.model_copy(update={"did": f"{self.__prefix}::{document.did}"}) for document in documents]
