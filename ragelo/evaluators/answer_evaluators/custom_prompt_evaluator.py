from ragelo.evaluators.answer_evaluators.base_answer_evaluator import AnswerEvaluatorFactory, BaseAnswerEvaluator
from ragelo.types.configurations import CustomPromptAnswerEvaluatorConfig
from ragelo.types.types import AnswerEvaluatorTypes


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.CUSTOM_PROMPT)
class CustomPromptEvaluator(BaseAnswerEvaluator):
    config: CustomPromptAnswerEvaluatorConfig
