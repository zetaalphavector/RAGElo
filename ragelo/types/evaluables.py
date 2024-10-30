from __future__ import annotations

from typing import Any

from ragelo.logger import logger
from ragelo.types.pydantic_models import BaseModel, validator
from ragelo.types.results import EvaluatorResult


class ChatMessage(BaseModel):
    sender: str
    content: str


class Evaluable(BaseModel):
    """A base class for objects that can be evaluated. Either a retrieved document or an agent answer.
    Args:
        evaluation Optional[EvaluatorResult]: The result of the evaluation.
        metadata Optional[dict[str, Any]]: Metadata that can be templated in the prompt.
    """

    evaluation: EvaluatorResult | None = None
    metadata: dict[str, Any] | None = None

    def add_metadata(self, metadata: dict[str, Any] | None):
        if not metadata:
            return
        if self.metadata is None:
            self.metadata = {}
        for k in metadata:
            if k in self.metadata:
                logger.warning(
                    f"Metadata {k} for {self.__class__.__name__}"
                    " is being overwritten!\n"
                    f"Old metadata: {self.metadata[k]}\n"
                    f"New metadata: {metadata[k]}\n"
                )
            self.metadata[k] = metadata[k]


class Document(Evaluable):
    """A document retrieved by an agent in response to a query.
    Args:
        did str: The document ID.
        text str: The text of the document.
        retrieved_by dict[str, float]: If a document was retrieved by multiple agents, the score attributed by each agent for this document.
    """

    did: str
    text: str
    retrieved_by: dict[str, float] = {}

    def add_retrieved_by(
        self, agent: str, score: float | None = None, overwrite: bool = False
    ):
        """Adds the score of an agent that retrieved the document."""
        if agent in self.retrieved_by and not overwrite:
            logger.info(
                f"Document with did {self.did} already retrieved by agent {agent}"
            )
            return
        if score is None:
            score = 1.0
        self.retrieved_by[agent] = score

    def __str__(self) -> str:
        return self.text


class AgentAnswer(Evaluable):
    """An answer generated by an agent in response to a query.
    Args:
        agent str: The agent that provided the answer.
        text str: The text of the answer.
        conversation Optional[list[ChatMessage]]: The conversation between the user and the agent.
    """

    agent: str
    text: str | None = None
    conversation: list[ChatMessage] | None = None

    @validator
    @classmethod
    def one_of_text_or_conversation(cls, values):
        text = values.get("text")
        conversation = values.get("conversation")
        if text is None and conversation is None:
            raise ValueError("Either text or conversation must be provided")

        if text is not None and conversation is not None:
            raise ValueError("Only one of text or conversation must be provided")
        return values


class PairwiseGame(Evaluable):
    """A game to be played between two agent answers.
    Args:
        agent_a_answer AgentAnswer: The answer provided by the first agent.
        agent_b_answer AgentAnswer: The answer provided by the second agent.
    """

    agent_a_answer: AgentAnswer
    agent_b_answer: AgentAnswer
