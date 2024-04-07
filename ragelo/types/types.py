from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


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


@dataclass
class AgentAnswer:
    agent: str
    text: str
    metadata: Optional[dict[str, Any]] = None


@dataclass
class Query:
    qid: str
    query: str
    metadata: Optional[dict[str, Any]] = None
    retrieved_docs: list[Document] = field(default_factory=list)
    answers: list[AgentAnswer] = field(default_factory=list)


@dataclass
class RetrievalEvaluatorResult:
    qid: str
    did: str
    raw_answer: str
    answer: Any


@dataclass
class AnswerEvaluatorResult:
    qid: str
    agent: str
    raw_answer: str
    answer: Any
