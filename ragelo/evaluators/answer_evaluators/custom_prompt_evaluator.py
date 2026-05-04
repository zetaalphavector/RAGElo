from ragelo.evaluators.answer_evaluators.base_answer_evaluator import AnswerEvaluatorFactory, BaseAnswerEvaluator
from ragelo.types.configurations import CustomPromptAnswerEvaluatorConfig
from ragelo.types.results import AnswerEvaluatorResult
from ragelo.types.types import AnswerEvaluatorTypes


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.CUSTOM_PROMPT)
class CustomPromptEvaluator(BaseAnswerEvaluator[CustomPromptAnswerEvaluatorConfig, AnswerEvaluatorResult]):
    config: CustomPromptAnswerEvaluatorConfig
    result_type = AnswerEvaluatorResult
