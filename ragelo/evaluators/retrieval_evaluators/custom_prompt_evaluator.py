from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import (
    CustomPromptEvaluatorConfig,
    RetrievalEvaluatorTypes,
)


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.CUSTOM_PROMPT)
class CustomPromptEvaluator(BaseRetrievalEvaluator):
    config: CustomPromptEvaluatorConfig

    def __init__(
        self,
        config: CustomPromptEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        self.__prompt = config.prompt

    def _build_message(self, query: str, document: str) -> str:
        formatters = {
            self.config.query_placeholder: query,
            self.config.document_placeholder: document,
        }

        return self.__prompt.format(**formatters)

    def _process_answer(self, answer: str) -> str:
        return self.json_answer_parser(answer, "relevance")
