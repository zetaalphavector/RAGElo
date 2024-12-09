from __future__ import annotations

from ragelo.evaluators.retrieval_evaluators import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import FewShotEvaluatorConfig
from ragelo.types.evaluables import Document
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

    def _build_message(self, query: Query, document: Document) -> list[dict[str, str]]:
        system_prompt_msg = {"role": "system", "content": self.system_prompt}
        messages = [system_prompt_msg] + self.__build_few_shot_samples()
        formatters = {
            self.config.query_placeholder: query.query,
            self.config.document_placeholder: document.text,
        }
        user_message = self.user_prompt.format(**formatters)

        messages.append({"role": "user", "content": user_message})
        return messages

    def __build_few_shot_samples(self) -> list[dict[str, str]]:
        few_shot_messages = []
        for few_shot in self.few_shots:
            formatters = {
                self.config.query_placeholder: few_shot.query,
                self.config.document_placeholder: few_shot.passage,
            }
            user_message = {
                "role": "user",
                "content": self.user_prompt.format(**formatters),
            }
            formatters = {
                self.config.reasoning_placeholder: few_shot.reasoning,
                self.config.relevance_placeholder: str(few_shot.relevance),
            }
            answer_text = self.assistant_prompt.format(**formatters)
            answer = {"role": "assistant", "content": answer_text}
            few_shot_messages.append(user_message)
            few_shot_messages.append(answer)
        return few_shot_messages
