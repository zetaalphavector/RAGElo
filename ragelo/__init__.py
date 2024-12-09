"""CLI for automatically evaluating Retrieval Augmented Generation (RAG) models."""

from __future__ import annotations

from ragelo.agent_rankers.base_agent_ranker import get_agent_ranker
from ragelo.evaluators.answer_evaluators import (
    AnswerEvaluatorFactory,
    get_answer_evaluator,
)
from ragelo.evaluators.retrieval_evaluators import (
    RetrievalEvaluatorFactory,
    get_retrieval_evaluator,
)
from ragelo.llm_providers.base_llm_provider import get_llm_provider
from ragelo.types import AgentAnswer, Document, Experiment, Query

__all__ = [
    "get_agent_ranker",
    "get_answer_evaluator",
    "get_retrieval_evaluator",
    "get_llm_provider",
    "AnswerEvaluatorFactory",
    "RetrievalEvaluatorFactory",
    "AgentAnswer",
    "Document",
    "Query",
    "Experiment",
]

__version__ = "0.1.0"
__app_name__ = "ragelo"
__author__ = "Zeta Alpha"
__email__ = "camara@zeta-alpha.com"
__description__ = "CLI for automatically evaluating Retrieval Augmented Generation (RAG) models."
