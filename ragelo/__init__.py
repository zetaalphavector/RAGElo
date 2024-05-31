"""CLI for automatically evaluating Retrieval Augmented Generation (RAG) models."""

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
from ragelo.types.types import AgentAnswer, Document, Query

__version__ = "0.1.0"
__app_name__ = "ragelo"
__author__ = "Zeta Alpha"
__email__ = "camara@zeta-alpha.com"
__description__ = (
    "CLI for automatically evaluating Retrieval Augmented Generation (RAG) models."
)
