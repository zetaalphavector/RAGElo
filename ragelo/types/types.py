from dataclasses import dataclass
from enum import Enum
from typing import Optional


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
class Query:
    qid: str
    query: str


@dataclass
class Document:
    did: str
    text: str
    query: Optional[Query] = None


@dataclass
class AgentAnswer:
    query: Query
    agent: str
    text: str
