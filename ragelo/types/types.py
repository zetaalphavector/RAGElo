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
    JEV = "jev"
    JEV_RDNAM = "jev_rdnam"
    JEV_RUBRIC_COVERAGE = "jev_rubric_coverage"
    REASONER = "reasoner"
    RUBRIC_COVERAGE = "rubric_coverage"


class LLMProviderTypes(StrEnum):
    """Enum that contains the names of the available LLM providers"""

    OPENAI = "openai"
    OLLAMA = "ollama"
    INSTRUCTOR = "instructor"
    VERCEL = "vercel"
    VERCEL_JEV = "vercel-jev"
    TYPESAFE = "typesafe"


class AnswerEvaluatorTypes(StrEnum):
    """Enum that contains the names of the available answer evaluators"""

    PAIRWISE = "pairwise"
    CUSTOM_PAIRWISE = "custom_pairwise"
    CUSTOM_PROMPT = "custom_prompt"
    DOMAIN_EXPERT = "domain_expert"
    CHAT_PAIRWISE = "chat_pairwise"
    JEV = "jev"
    JEV_PAIRWISE = "jev_pairwise"
    JEV_RUBRIC_PAIRWISE = "jev_rubric_pairwise"
    JEV_RUBRIC_POINTWISE = "jev_rubric_pointwise"
    RUBRIC_PAIRWISE = "rubric_pairwise"
    RUBRIC_POINTWISE = "rubric_pointwise"


class AgentRankerTypes(StrEnum):
    ELO = "elo"
