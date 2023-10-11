"""CLI for automatically evaluating answers of Retrieval Augmented Generation (RAG) models."""
from loguru import logger

# disable logger if used as a library by default
logger.disable("auto_eval")

__version__ = "0.1.0"
__app_name__ = "auto-eval"
__author__ = "Zeta Alpha"
__email__ = "camara@zeta-alpha.com"
__description__ = "CLI for automatically evaluating answers of Retrieval Augmented Generation (RAG) models."
