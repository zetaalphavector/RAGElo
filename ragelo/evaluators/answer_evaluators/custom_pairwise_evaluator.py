from ragelo.evaluators.answer_evaluators.base_answer_evaluator import AnswerEvaluatorFactory
from ragelo.evaluators.answer_evaluators.pairwise_evaluator import PairwiseAnswerEvaluator
from ragelo.types.configurations import CustomPairwiseEvaluatorConfig
from ragelo.types.types import AnswerEvaluatorTypes


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.CUSTOM_PAIRWISE)
class CustomPairwiseEvaluator(PairwiseAnswerEvaluator):
    """A custom pairwise evaluator that allows for additional customization."""

    config: CustomPairwiseEvaluatorConfig
