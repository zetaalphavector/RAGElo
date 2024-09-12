"""CLI for automatically evaluating Retrieval Augmented Generation (RAG) models."""

from ragelo.agent_rankers.base_agent_ranker import get_agent_ranker
from ragelo.evaluators.answer_evaluators import get_answer_evaluator
from ragelo.evaluators.retrieval_evaluators import get_retrieval_evaluator
from ragelo.llm_providers.base_llm_provider import get_llm_provider
from ragelo.types import AgentAnswer, Document, PairwiseGame, Query

__all__ = [
    "AgentAnswer",
    "Document",
    "Query",
    "PairwiseGame",
    "get_agent_ranker",
    "get_answer_evaluator",
    "get_retrieval_evaluator",
    "get_llm_provider",
]
__version__ = "0.1.0"
__app_name__ = "ragelo"
__author__ = "Zeta Alpha"
__email__ = "camara@zeta-alpha.com"
__description__ = "CLI for automatically evaluating Retrieval Augmented Generation (RAG) models."
