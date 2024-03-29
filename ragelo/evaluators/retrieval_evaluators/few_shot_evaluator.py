from ragelo.evaluators.retrieval_evaluators import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import Document, RetrievalEvaluatorTypes
from ragelo.types.configurations import FewShotEvaluatorConfig


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.FEW_SHOT)
class FewShotEvaluator(BaseRetrievalEvaluator):
    config: FewShotEvaluatorConfig

    def __init__(
        self,
        config: FewShotEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)

        self.prompt = config.prompt
        self.sys_prompt = config.system_prompt
        self.few_shots = config.few_shots

    def _build_message(self, document: Document) -> list[dict[str, str]]:
        system_prompt_msg = {"role": "system", "content": self.sys_prompt}
        messages = [system_prompt_msg] + self.__build_few_shot_samples()
        user_message = self.prompt.format(
            query=document.query.query, passage=document.text
        )
        messages.append({"role": "user", "content": user_message})
        return messages

    def __build_few_shot_samples(self) -> list[dict[str, str]]:
        few_shot_messages = []
        for few_shot in self.few_shots:
            user_message = {
                "role": "user",
                "content": self.prompt.format(
                    query=few_shot.query, passage=few_shot.passage
                ),
            }
            answer = {"role": "assistant", "content": few_shot.reasoning}
            few_shot_messages.append(user_message)
            few_shot_messages.append(answer)
        return few_shot_messages

    def _process_answer(self, answer: str) -> str:
        return self.json_answer_parser(answer, "relevance")
