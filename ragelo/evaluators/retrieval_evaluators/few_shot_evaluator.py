from __future__ import annotations

from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator, RetrievalEvaluatorFactory
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import FewShotEvaluatorConfig
from ragelo.types.evaluables import Document
from ragelo.types.formats import LLMInputPrompt
from ragelo.types.query import Query
from ragelo.types.types import RetrievalEvaluatorTypes
from ragelo.utils import string_to_template


@RetrievalEvaluatorFactory.register(RetrievalEvaluatorTypes.FEW_SHOT)
class FewShotEvaluator(BaseRetrievalEvaluator):
    config: FewShotEvaluatorConfig
    user_prompt = string_to_template("""
        Query: {{ query.query }}
        Passage: {{ document.text }}""")
    few_shot_assistant_answer = string_to_template("""{"reasoning": "{{reasoning}}", "relevance": {{relevance}}}""")

    def __init__(self, config: FewShotEvaluatorConfig, llm_provider: BaseLLMProvider):
        super().__init__(config, llm_provider)
        if config.user_prompt:
            self.user_prompt = config.user_prompt
        if config.few_shot_assistant_answer:
            self.few_shot_assistant_answer = config.few_shot_assistant_answer
        if config.system_prompt:
            self.system_prompt = config.system_prompt
        self.few_shots = config.few_shots

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        few_shot_messages = self.__build_few_shot_examples()
        few_shot_messages.append({"role": "user", "content": self.user_prompt.render(query=query, document=document)})
        if self.system_prompt:
            return LLMInputPrompt(
                system_prompt=self.system_prompt.render(query=query, document=document), messages=few_shot_messages
            )

        return LLMInputPrompt(messages=few_shot_messages)

    def __build_few_shot_examples(self) -> list[dict[str, str]]:
        few_shot_messages: list[dict[str, str]] = []
        for idx, few_shot in enumerate(self.few_shots):
            query = Query(query=few_shot.query, qid=f"query_{idx}")
            document = Document(text=few_shot.passage, did=f"doc_{idx}", qid=f"query_{idx}")
            few_shot_messages.append(
                {"role": "user", "content": self.user_prompt.render(query=query, document=document)}
            )
            reasoning = few_shot.reasoning
            relevance = few_shot.relevance
            assistant_message = self.few_shot_assistant_answer.render(reasoning=reasoning, relevance=relevance)
            few_shot_messages.append({"role": "assistant", "content": assistant_message})
        return few_shot_messages
