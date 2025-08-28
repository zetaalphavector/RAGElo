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
        # Build few-shot transcript as plain text to be passed as user_message
        few_shots_text = self.__build_few_shot_transcript()
        formatters = {
            self.config.query_placeholder: query.query,
            self.config.document_placeholder: document.text,
        }
        user_message = f"{few_shots_text}User: {self.user_prompt.format(**formatters)}"

        return LLMInputPrompt(
            user_message=user_message,
            system_prompt=self.system_prompt,
        )

    def __build_few_shot_transcript(self) -> str:
        transcript_parts: list[str] = []
        for few_shot in self.few_shots:
            user_formatters = {
                self.config.query_placeholder: few_shot.query,
                self.config.document_placeholder: few_shot.passage,
            }
            assistant_formatters = {
                self.config.reasoning_placeholder: few_shot.reasoning,
                self.config.relevance_placeholder: str(few_shot.relevance),
            }
            transcript_parts.append(f"User: {self.user_prompt.format(**user_formatters)}")
            transcript_parts.append(f"Assistant: {self.assistant_prompt.format(**assistant_formatters)}")
        if transcript_parts:
            return "\n".join(transcript_parts) + "\n"
        return ""
