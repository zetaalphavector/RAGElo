from ragelo.evaluators.retrieval_evaluators import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import FewShotEvaluatorConfig
from ragelo.types.evaluables import Document
from ragelo.types.formats import LLMInputPrompt
from ragelo.types.query import Query
from ragelo.types.types import RetrievalEvaluatorTypes


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.FEW_SHOT)
class FewShotEvaluator(BaseRetrievalEvaluator):
    config: FewShotEvaluatorConfig

    def __init__(
        self,
        config: FewShotEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)

        self.user_prompt = config.few_shot_user_prompt
        self.system_prompt = config.system_prompt
        self.assistant_prompt = config.few_shot_assistant_answer
        self.few_shots = config.few_shots

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        few_shot_messages = self.__build_few_shot_examples()
        few_shot_messages.append({"role": "user", "content": self.user_prompt.render(query=query, document=document)})

        return LLMInputPrompt(
            system_prompt=self.system_prompt,
            messages=few_shot_messages,
        )

    def __build_few_shot_examples(self) -> list[dict[str, str]]:
        few_shot_messages: list[dict[str, str]] = []
        for few_shot in self.few_shots:
            query = Query(query=few_shot.query)
            document = Document(text=few_shot.passage)
            few_shot_messages.append(
                {"role": "user", "content": self.user_prompt.render(query=query, document=document)}
            )
            reasoning = few_shot.reasoning
            relevance = few_shot.relevance
            assistant_message = self.assistant_prompt.render(reasoning=reasoning, relevance=relevance)
            few_shot_messages.append({"role": "assistant", "content": assistant_message})
        return few_shot_messages
