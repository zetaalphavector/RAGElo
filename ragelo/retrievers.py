"""The contract between an Experiment and the retrieval systems it compares."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel

from ragelo.types.query import Query


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
