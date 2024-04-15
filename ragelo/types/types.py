from enum import StrEnum
from typing import Any, Optional

from ragelo.logger import logger
from ragelo.pydantic_v1 import _PYDANTIC_MAJOR_VERSION
from ragelo.pydantic_v1 import BaseModel as PydanticBaseModel


class AnswerFormat(StrEnum):
    """Enum that contains the names of the available answer formats"""

    JSON = "json"
    TEXT = "text"
    MULTI_FIELD_JSON = "multi_field_json"


class RetrievalEvaluatorTypes(StrEnum):
    """Enum that contains the names of the available retrieval evaluators"""

    CUSTOM_PROMPT = "custom_prompt"
    DOMAIN_EXPERT = "domain_expert"
    FEW_SHOT = "few_shot"
    RDNAM = "RDNAM"
    REASONER = "reasoner"


class LLMProviderTypes(StrEnum):
    """Enum that contains the names of the available LLM providers"""

    OPENAI = "openai"


class AnswerEvaluatorTypes(StrEnum):
    """Enum that contains the names of the available answer evaluators"""

    PAIRWISE_REASONING = "pairwise_reasoning"
    CUSTOM_PROMPT = "custom_prompt"


class BaseModel(PydanticBaseModel):
    def model_dump(self):
        if _PYDANTIC_MAJOR_VERSION == 1:
            return self.dict()  # type: ignore
        else:
            return super().model_dump()  # type: ignore


class FewShotExample(BaseModel):
    """A few-shot example."""

    passage: str
    query: str
    relevance: int
    reasoning: str


class Document(BaseModel):
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


class AgentAnswer(BaseModel):
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


class Query(BaseModel):
    qid: str
    query: str
    metadata: Optional[dict[str, Any]] = None
    retrieved_docs: list[Document] = []
    answers: list[AgentAnswer] = []

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


class RetrievalEvaluatorResult(BaseModel):
    qid: str
    did: str
    raw_answer: str
    answer: str | int | dict[str, Any]


class AnswerEvaluatorResult(BaseModel):
    qid: str
    raw_answer: str
    answer: str | int | dict[str, Any]
    agent: Optional[str] = None
    agent_a: Optional[str] = None
    agent_b: Optional[str] = None
