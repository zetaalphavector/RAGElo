from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    # Implementing __str__ to allow for easy conversion to string
    # e.g. str(RetrievalEvaluatorTypes.CUSTOM_PROMPT) -> "custom_prompt"
    # otherwise, it's "RetrievalEvaluatorTypes.CUSTOM_PROMPT"
    def __str__(self):
        return self.value


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
    OLLAMA = "ollama"


class AnswerEvaluatorTypes(StrEnum):
    """Enum that contains the names of the available answer evaluators"""

    PAIRWISE = "pairwise"
    CUSTOM_PAIRWISE = "custom_pairwise"
    CUSTOM_PROMPT = "custom_prompt"
    DOMAIN_EXPERT = "domain_expert"
    CHAT_PAIRWISE = "chat_pairwise"


class AgentRankerTypes(StrEnum):
    ELO = "elo"
