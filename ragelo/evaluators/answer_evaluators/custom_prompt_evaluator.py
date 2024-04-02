from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
    BaseAnswerEvaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import AgentAnswer, AnswerEvaluatorTypes
from ragelo.types.configurations import CustomPromptAnswerEvaluatorConfig


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.CUSTOM_PROMPT)
class CustomPromptEvaluator(BaseAnswerEvaluator):
    config: CustomPromptAnswerEvaluatorConfig
    scoring_key: str = "relevance"

    def __init__(
        self,
        config: CustomPromptAnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        self.__prompt = config.prompt

    def _build_message(self, answer: AgentAnswer) -> str:
        formatters = {
            self.config.query_placeholder: answer.query.query,
            self.config.answer_placeholder: answer.text,
        }

        return self.__prompt.format(**formatters)

    def _process_answer(self, answer: str) -> str:
        return self.json_answer_parser(answer, self.scoring_key)
