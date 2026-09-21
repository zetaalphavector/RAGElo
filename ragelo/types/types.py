from enum import Enum


class StrEnum(str, Enum):
    def __str__(self):
        return self.value


result_type_registry: dict[str, type] = {}


class RetrievalEvaluatorTypes(StrEnum):
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
    OPENAI = "openai"
    OLLAMA = "ollama"
    INSTRUCTOR = "instructor"
    VERCEL = "vercel"
    VERCEL_JEV = "vercel-jev"
    TYPESAFE = "typesafe"


class AnswerEvaluatorTypes(StrEnum):
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


class BenchmarkDatasetTypes(StrEnum):
    LLMJUDGE = "llmjudge"
    LLMJUDGE_PAIRWISE = "llmjudge_pairwise"
    TREC_RAG24 = "trec_rag24"
    TREC_RAG24_ANSWERS = "trec_rag24_answers"
