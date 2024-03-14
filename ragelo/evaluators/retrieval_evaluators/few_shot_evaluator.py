import json
from typing import Dict, List

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import Document, Query
from ragelo.types.configurations import FewShotEvaluatorConfig


@RetrievalEvaluatorFactory.register("few_shot")
class FewShotEvaluator(BaseRetrievalEvaluator):
    def __init__(
        self,
        config: FewShotEvaluatorConfig,
        queries: Dict[str, Query],
        documents: Dict[str, Dict[str, Document]],
        llm_provider: BaseLLMProvider,
    ):
        if not queries:
            raise ValueError(
                "You are trying to use a Retrieval Evaluator without providing queries"
            )
        if not documents:
            raise ValueError(
                "You are trying to use a Retrieval Evaluator without providing documents"
            )

        self.queries = queries
        self.documents = documents
        if not config.output_file:
            self.output_file = "retrieval_evaluator.log"
        else:
            self.output_file = config.output_file
        self.llm_provider = llm_provider
        self.config = config

        self.prompt = config.prompt
        self.sys_prompt = config.system_prompt
        self.few_shots = config.few_shots

    def _build_message(self, qid: str, did: str) -> List[Dict[str, str]]:
        query = self.queries[qid].query
        document = self.documents[qid][did].text
        system_prompt_msg = {"role": "system", "content": self.sys_prompt}
        messages = [system_prompt_msg] + self.__build_few_shot_samples()
        user_message = self.prompt.format(query=query, passage=document)
        messages.append({"role": "user", "content": user_message})
        return messages

    def __build_few_shot_samples(self) -> List[Dict[str, str]]:
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
        try:
            answer = json.loads(answer.strip().split("\n")[-1])["relevance"]
        except KeyError:
            raise ValueError(f"Answer {answer} does not contain a relevance field")
        return answer
