from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
    BaseAnswerEvaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import CustomPairwiseEvaluatorConfig
from ragelo.types.types import AnswerEvaluatorTypes


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.CUSTOM_PAIRWISE)
class CustomPairwiseEvaluator(BaseAnswerEvaluator):
    """A custom pairwise evaluator that allows for additional customization."""

    config: CustomPairwiseEvaluatorConfig

    def __init__(self, config: CustomPairwiseEvaluatorConfig, llm_provider: BaseLLMProvider):
        super().__init__(config, llm_provider)
        self.system_prompt = config.system_prompt
        self.user_prompt = config.user_prompt
