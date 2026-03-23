from enum import Enum


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


# Global registry mapping evaluator names to their result types.
# Populated by factory @register decorators so that result-type resolution
# does not need to import the evaluator packages.
_result_type_registry: dict[str, type] = {}


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
    RUBRIC_PAIRWISE = "rubric_pairwise"
    RUBRIC_POINTWISE = "rubric_pointwise"


class AgentRankerTypes(StrEnum):
    ELO = "elo"
