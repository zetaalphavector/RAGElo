from enum import Enum
from importlib import metadata
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel as PydanticBaseModel

from ragelo.logger import logger

_PYDANTIC_MAJOR_VERSION: int = int(metadata.version("pydantic").split(".")[0])
if _PYDANTIC_MAJOR_VERSION == 1:
    from pydantic import root_validator

    validator = root_validator(pre=True)  # type: ignore
    ValidationError = TypeError
else:
    from pydantic import ValidationError  # type: ignore
    from pydantic import model_validator  # type: ignore

    validator = model_validator(mode="before")  # type: ignore


class BaseModel(PydanticBaseModel):
    @classmethod
    def get_model_fields(cls):
        if _PYDANTIC_MAJOR_VERSION == 1:
            return cls.__fields__  # type: ignore
        else:
            return cls.model_fields  # type: ignore

    def model_dump(self):
        if _PYDANTIC_MAJOR_VERSION == 1:
            return self.dict()  # type: ignore
        else:
            return super().model_dump()  # type: ignore


class AnswerFormat(str, Enum):
    """Enum that contains the names of the available answer formats"""

    JSON = "json"
    TEXT = "text"
    MULTI_FIELD_JSON = "multi_field_json"


class RetrievalEvaluatorTypes(str, Enum):
    """Enum that contains the names of the available retrieval evaluators"""

    CUSTOM_PROMPT = "custom_prompt"
    DOMAIN_EXPERT = "domain_expert"
    FEW_SHOT = "few_shot"
    RDNAM = "RDNAM"
    REASONER = "reasoner"


class LLMProviderTypes(str, Enum):
    """Enum that contains the names of the available LLM providers"""

    OPENAI = "openai"


class AnswerEvaluatorTypes(str, Enum):
    """Enum that contains the names of the available answer evaluators"""

    PAIRWISE = "pairwise"
    CUSTOM_PROMPT = "custom_prompt"
    DOMAIN_EXPERT = "domain_expert"


class EvaluatorResult(BaseModel):
    """Generic class with the results of an evaluator.
    Args:
        qid str: The query ID to which the result corresponds.
        raw_answer Optional[str]: The raw answer provided by the LLMProvider used in the evaluator.
        answer Optional[Union[int, str, Dict[str, Any]]]: The processed answer provided by the evaluator.
            If the evaluator uses "multi_field_json" as answer_format, this will be a dictionary with evaluation keys.
    """

    qid: str
    raw_answer: Optional[str]
    answer: Optional[Union[int, str, Dict[str, Any]]]
    exception: Optional[str] = None

    @validator
    @classmethod
    def check_agents(cls, v):
        exception = v.get("exception")
        raw_answer = v.get("raw_answer")
        answer = v.get("answer")
        if (raw_answer is None or answer is None) and exception is None:
            raise ValidationError(
                "Either answer or raw_answer must be provided. Otherwise, an exception must be provided."
            )  # type: ignore
        return v


class AnswerEvaluatorResult(EvaluatorResult):
    """The results of an answer evaluator.
    Args:
        did str: The document ID to which the result corresponds.
        agent Optional[str]: The agent that provided the answer. Only used if pairwise=False.
        agent_a Optional[str]: The first agent that provided the answer. Only used if the evaluator is pairwise.
        agent_b Optional[str]: The second agent that provided the answer. Only used if the evaluator is pairwise.
        pairwise bool: Whether the evaluation is pairwise or not.
    """

    agent: Optional[str] = None
    agent_a: Optional[str] = None
    agent_b: Optional[str] = None
    pairwise: bool = False

    @validator
    @classmethod
    def check_agents(cls, v):
        agent = v.get("agent")
        agent_a = v.get("agent_a")
        agent_b = v.get("agent_b")
        if agent is None and agent_a is None and agent_b is None:
            raise ValidationError(
                "Either agent or agent_a and agent_b must be provided"
            )  # type: ignore
        if agent_a is not None and agent_b is not None:
            v["pairwise"] = True
        return v


class RetrievalEvaluatorResult(EvaluatorResult):
    """The results of a retrieval evaluator.
    Args:
        did str: The document ID to which the result corresponds.
    """

    did: str


class FewShotExample(BaseModel):
    """A few-shot example used in the few-shot retrieval evaluator.
    Args:
        passage str: The passage of the example.
        query str: The query of the example.
        relevance int: The relevance of the example.
        reasoning str: The reasoning behind the relevance
    """

    passage: str
    query: str
    relevance: int
    reasoning: str


class Evaluable(BaseModel):
    """A base class for objects that can be evaluated. Either a document or an agent answer.
    Args:
        evaluation Optional[EvaluatorResult]: The result of the evaluation.
        metadata Optional[Dict[str, Any]]: Metadata that can be templated in the prompt.
    """

    evaluation: Optional[EvaluatorResult] = None
    metadata: Optional[Dict[str, Any]] = None

    def add_metadata(self, metadata: Optional[Dict[str, Any]]):
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
        score Optional[float]: The score of the document when retrieved.
        agent Optional[str]: The agent that retrieved the document.
    """

    did: str
    text: str
    score: Optional[float] = None
    agent: Optional[str] = None


class AgentAnswer(Evaluable):
    """An answer generated by an agent in response to a query.
    Args:
        agent str: The agent that provided the answer.
        text str: The text of the answer.
    """

    agent: str
    text: str


class PairwiseGame(Evaluable):
    """A game to be played between two agent answers.
    Args:
        agent_a_answer AgentAnswer: The answer provided by the first agent.
        agent_b_answer AgentAnswer: The answer provided by the second agent.
    """

    agent_a_answer: AgentAnswer
    agent_b_answer: AgentAnswer


class Query(BaseModel):
    """A user query that can have retrieved documents and agent answers.
    Args:
        qid str: The query ID.
        query str: The query text.
        metadata Optional[Dict[str, Any]]: Metadata that can be templated in the prompt.
        retrieved_docs List[Document]: The list of retrieved documents.
        answers List[AgentAnswer]: The list of agent answers.
        pairwise_games List[PairwiseGame]: The list games to be played between agent's answers. Generated by the evaluator.
    """

    qid: str
    query: str
    metadata: Optional[Dict[str, Any]] = None
    retrieved_docs: List[Document] = []
    answers: List[AgentAnswer] = []
    pairwise_games: List[PairwiseGame] = []

    def add_metadata(self, metadata: Optional[Dict[str, Any]]):
        """Adds metadata to the query that may be templated in the prompt.
        Args:
            metadata Dict[str, Any]: The metadata to add to the query.
        """
        if not metadata:
            return
        if self.metadata is None:
            self.metadata = {}
        for k in metadata:
            if k in self.metadata:
                logger.warning(
                    f"Metadata {k} for query {self.qid} is being overwritten!\n"
                    f"Old metadata: {self.metadata[k]}\n"
                    f"New metadata: {metadata[k]}\n"
                )
            self.metadata[k] = metadata[k]

    def add_retrieved_doc(
        self,
        doc: Union[Document, str],
        doc_id: Optional[str] = None,
        score: Optional[float] = None,
        agent: Optional[str] = None,
    ):
        """Add a retrieved document to the query.
        Args:
            doc Document | str: The document (either the object or the text) to add to the query.
            doc_id str: The document ID. If not provided, it will be autogenerated.
            score float: The score of the document when retrieved.
            agent str: The agent that retrieved the document.
        """
        if isinstance(doc, str):
            if doc_id is None:
                existing_docs = len(self.retrieved_docs)
                did = existing_docs + 1
                logger.info(f"No doc_id provided. Using '<doc_{did}>' as doc_id")
                doc_id = f"doc_{did}"
            doc = Document(did=doc_id, text=doc, score=score, agent=agent)
        existing_dids = [d.did for d in self.retrieved_docs]
        if doc.did in existing_dids:
            logger.info(
                f"Document with did {doc.did} already exists in query {self.qid}"
            )
            return
        self.retrieved_docs.append(doc)

    def add_agent_answer(
        self, answer: Union[AgentAnswer, str], agent: Optional[str] = None
    ):
        """Add an answer generated by an agent to the query.
        Args:
            answer AgentAnswer | str: The answer (either the object or the text) to add to the query.
            agent str: The id or name of the agent that generated the answer. If not provided, it will be autogenerated.
        """
        if isinstance(answer, str):
            if agent is None:
                agent = f"agent_{len(self.answers) + 1}"
                logger.info(f"No agent provided. Using '{agent}' as agent")
            answer = AgentAnswer(agent=agent, text=answer)
        existing_agents = [a.agent for a in self.answers]
        if answer.agent in existing_agents:
            logger.warning(
                f"Answer from agent {answer.agent} already exists in query {self.qid}"
            )
            return
        self.answers.append(answer)
