from ragelo.evaluators.answer_evaluators.base_answer_evaluator import AnswerEvaluatorFactory, BaseAnswerEvaluator
from ragelo.types.configurations import CustomPairwiseEvaluatorConfig
from ragelo.types.types import AnswerEvaluatorTypes


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.CUSTOM_PAIRWISE)
class CustomPairwiseEvaluator(BaseAnswerEvaluator):
    """A custom pairwise evaluator that allows for additional customization."""

    config: CustomPairwiseEvaluatorConfig
