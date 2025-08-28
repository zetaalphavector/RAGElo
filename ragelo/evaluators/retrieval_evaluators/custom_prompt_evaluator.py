from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import CustomPromptEvaluatorConfig
from ragelo.types.evaluables import Document
from ragelo.types.formats import LLMInputPrompt
from ragelo.types.query import Query
from ragelo.types.types import RetrievalEvaluatorTypes


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.CUSTOM_PROMPT)
class CustomPromptEvaluator(BaseRetrievalEvaluator):
    config: CustomPromptEvaluatorConfig

    def __init__(
        self,
        config: CustomPromptEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        self.system_prompt = config.system_prompt
        self.user_prompt = config.user_prompt

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        context = {"query": query, "document": document}
        user_message = self.user_prompt.render(**context) if self.user_prompt else None
        system_prompt = self.system_prompt.render(**context) if self.system_prompt else None

        return LLMInputPrompt(
            system_prompt=system_prompt,
            user_message=user_message,
        )
