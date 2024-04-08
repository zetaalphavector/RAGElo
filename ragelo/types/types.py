from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ragelo.logger import logger


class RetrievalEvaluatorTypes(str, Enum):
    """Enum that contains the names of the available retrieval evaluators"""

    CUSTOM_PROMPT = "custom_prompt"
    DOMAIN_EXPERT = "domain_expert"
    FEW_SHOT = "few_shot"
    RDNAM = "RDNAM"
    REASONER = "reasoner"


class LLMProviderTypes(str, Enum):
    """Enum that contains the names of the available LLM providers"""

    OPENAI = "openai"


class AnswerEvaluatorTypes(str, Enum):
    """Enum that contains the names of the available answer evaluators"""

    PAIRWISE_REASONING = "pairwise_reasoning"
    CUSTOM_PROMPT = "custom_prompt"


@dataclass
class Document:
    did: str
    text: str
    metadata: Optional[dict[str, Any]] = None

    def add_metadata(self, metadata: Optional[dict[str, Any]]):
        if not metadata:
            return
        if self.metadata is None:
            self.metadata = {}
        for k in metadata:
            if k in self.metadata:
                logger.warning(
                    f"Metadata {k} for document {self.did}"
                    " is being overwritten!\n"
                    f"Old metadata: {self.metadata[k]}\n"
                    f"New metadata: {metadata[k]}\n"
                )
            self.metadata[k] = metadata[k]


@dataclass
class AgentAnswer:
    agent: str
    text: str
    metadata: Optional[dict[str, Any]] = None

    def add_metadata(self, metadata: Optional[dict[str, Any]]):
        if not metadata:
            return
        if self.metadata is None:
            self.metadata = {}
        for k in metadata:
            if k in self.metadata:
                logger.warning(
                    f"Metadata {k} for answer from agent {self.agent}"
                    " is being overwritten!\n"
                    f"Old metadata: {self.metadata[k]}\n"
                    f"New metadata: {metadata[k]}\n"
                )
            self.metadata[k] = metadata[k]


@dataclass
class Query:
    qid: str
    query: str
    metadata: Optional[dict[str, Any]] = None
    retrieved_docs: list[Document] = field(default_factory=list)
    answers: list[AgentAnswer] = field(default_factory=list)

    def add_metadata(self, metadata: Optional[dict[str, Any]]):
        if not metadata:
            return
        if self.metadata is None:
            self.metadata = {}
        for k in metadata:
            if k in self.metadata:
                logger.warning(
                    f"Metadata {k} for query {self.qid} is being overwritten!\n"
                    f"Old metadata: {self.metadata[k]}\n"
                    f"New metadata: {metadata[k]}\n"
                )
            self.metadata[k] = metadata[k]


@dataclass
class RetrievalEvaluatorResult:
    qid: str
    did: str
    raw_answer: str
    answer: str | int | dict[str, Any]


@dataclass
class AnswerEvaluatorResult:
    qid: str
    raw_answer: str
    answer: str | int | dict[str, Any]
    agent: Optional[str] = None
    agent_a: Optional[str] = None
    agent_b: Optional[str] = None
