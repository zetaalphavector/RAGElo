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
    output_file: str = "custom_prompt_evaluations.csv"
    output_columns = ["query_id", "agent", "raw_answer"]

    def __init__(
        self,
        config: CustomPromptAnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        self.__prompt = config.prompt
        self.__scoring_fields = config.scoring_fields
        self.output_columns.extend(self.__scoring_fields)

    def _build_message(self, answer: AgentAnswer) -> str:
        reasonings = self._prepare_reasonings(answer.query.qid)
        formatters = {
            self.config.query_placeholder: answer.query.query,
            self.config.answer_placeholder: answer.text,
            self.config.documents_placeholder: reasonings,
        }

        return self.__prompt.format(**formatters)

    def _process_answer(self, answer: str) -> dict[str, str]:
        return self.json_answer_parser_multifields(answer, self.__scoring_fields)
