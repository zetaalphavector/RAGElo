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
            )
        return v


class AnswerEvaluatorResult(EvaluatorResult):
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
            )
        if agent_a is not None and agent_b is not None:
            v["pairwise"] = True
        return v


class RetrievalEvaluatorResult(EvaluatorResult):
    did: str


class FewShotExample(BaseModel):
    """A few-shot example."""

    passage: str
    query: str
    relevance: int
    reasoning: str


class Evaluable(BaseModel):
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
    did: str
    text: str


class AgentAnswer(Evaluable):
    agent: str
    text: str


class PairwiseGame(Evaluable):
    agent_a_answer: AgentAnswer
    agent_b_answer: AgentAnswer


class Query(BaseModel):
    qid: str
    query: str
    metadata: Optional[Dict[str, Any]] = None
    retrieved_docs: List[Document] = []
    answers: List[AgentAnswer] = []
    pairwise_games: List[PairwiseGame] = []

    def add_metadata(self, metadata: Optional[Dict[str, Any]]):
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
        self, doc: Union[Document, str], doc_id: Optional[str] = None
    ):
        if isinstance(doc, str):
            if doc_id is None:
                raise ValueError("doc_id must be provided if doc is a string")
            doc = Document(did=doc_id, text=doc)
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
        if isinstance(answer, str):
            if agent is None:
                raise ValueError("agent must be provided if answer is a string")
            answer = AgentAnswer(agent=agent, text=answer)
        existing_agents = [a.agent for a in self.answers]
        if answer.agent in existing_agents:
            logger.warning(
                f"Answer from agent {answer.agent} already exists in query {self.qid}"
            )
            return
        self.answers.append(answer)
