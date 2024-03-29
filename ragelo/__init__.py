"""CLI for automatically evaluating Retrieval Augmented Generation (RAG) models."""

from ragelo.evaluators.answer_evaluators import (
    AnswerEvaluatorFactory,
    get_answer_evaluator,
)
from ragelo.evaluators.retrieval_evaluators import (
    RetrievalEvaluatorFactory,
    get_retrieval_evaluator,
)
from ragelo.llm_providers.base_llm_provider import get_llm_provider

__version__ = "0.1.0"
__app_name__ = "ragelo"
__author__ = "Zeta Alpha"
__email__ = "camara@zeta-alpha.com"
__description__ = (
    "CLI for automatically evaluating Retrieval Augmented Generation (RAG) models."
)
