from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
    BaseAnswerEvaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import CustomPromptAnswerEvaluatorConfig
from ragelo.types.evaluables import AgentAnswer
from ragelo.types.formats import LLMInputPrompt
from ragelo.types.query import Query
from ragelo.types.types import AnswerEvaluatorTypes


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.CUSTOM_PROMPT)
class CustomPromptEvaluator(BaseAnswerEvaluator):
    config: CustomPromptAnswerEvaluatorConfig

    def __init__(
        self,
        config: CustomPromptAnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        self.system_prompt = config.system_prompt
        self.user_prompt = config.user_prompt

    def _build_message(self, query: Query, answer: AgentAnswer) -> LLMInputPrompt:
        documents = self._filter_documents(query)
        context = {"query": query, "answer": answer, "documents": documents}
        user_message = self.user_prompt.render(**context)
        system_prompt = self.system_prompt.render(**context) if self.system_prompt else None
        return LLMInputPrompt(
            system_prompt=system_prompt,
            user_message=user_message,
        )
