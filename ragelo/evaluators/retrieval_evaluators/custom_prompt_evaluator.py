from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.types.configurations import CustomPromptEvaluatorConfig
from ragelo.types.types import RetrievalEvaluatorTypes


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.CUSTOM_PROMPT)
class CustomPromptEvaluator(BaseRetrievalEvaluator):
    config: CustomPromptEvaluatorConfig
