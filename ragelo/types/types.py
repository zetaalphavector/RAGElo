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
    OLLAMA = "ollama"


class AnswerEvaluatorTypes(str, Enum):
    """Enum that contains the names of the available answer evaluators"""

    PAIRWISE = "pairwise"
    CUSTOM_PROMPT = "custom_prompt"
    DOMAIN_EXPERT = "domain_expert"


class AgentRankerTypes(str, Enum):
    ELO = "elo"


class EvaluatorResult(BaseModel):
    """Generic class with the results of an evaluator.
    Args:
        qid str: The query ID to which the result corresponds.
        agent Optional[str]: The agent that provided the answer or retrieved the document.
        raw_answer Optional[str]: The raw answer provided by the LLMProvider used in the evaluator.
        answer Optional[Union[int, str, Dict[str, Any]]]: The processed answer provided by the evaluator.
            If the evaluator uses "multi_field_json" as answer_format, this will be a dictionary with evaluation keys.
    """

    qid: str
    agent: Optional[str] = None
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


class Document(Evaluable):
    """A document retrieved by an agent in response to a query.
    Args:
        did str: The document ID.
        text str: The text of the document.
        retrieved_by Dict[str, float]: If a document was retrieved by multiple agents, the score attributed by each agent for this document.
    """

    did: str
    text: str
    retrieved_by: Dict[str, float] = {}

    def add_retrieved_by(
        self, agent: str, score: Optional[float] = None, overwrite: bool = False
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


class Query(BaseModel):
    """A user query that can have retrieved documents and agent answers.
    Args:
        qid str: The query ID.
        query str: The query text.
        metadata Optional[Dict[str, Any]]: Metadata that can be templated in the prompt.
        retrieved_docs Dict[str, Document]: A dictionary of retrieved documents, where the key is the document ID.
        answers List[AgentAnswer]: The list of agent answers.
        pairwise_games List[PairwiseGame]: The list games to be played between agent's answers. Generated by the evaluator.
    """

    qid: str
    query: str
    metadata: Optional[Dict[str, Any]] = None
    retrieved_docs: Dict[str, Document] = {}
    answers: Dict[str, AgentAnswer] = {}
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

    def add_retrieved_docs(
        self,
        docs: List,
        agent: Optional[str] = None,
        overwrite: bool = False,
    ):
        """Add a list of retrieved documents to the query.
        Args:
            docs List[Document | str] | List[Tuple[Document | str], float]: The list of documents (either the objects or the texts) to add to the query. Optionally, a score can be provided for each document.
            agent str: The agent that retrieved the documents.
        """
        # First, transform all of the strings into Document objects.
        # If an agent name was provided, but no score information exists,
        # assume the list is ranked and assign a score of 1/(n+1) to each document.
        for idx, doc in enumerate(docs):
            if isinstance(doc, tuple):
                doc, score = doc
                if agent is None:
                    raise ValueError(
                        "If you provide a score for each retrieved document, you should also provide an agent name."
                    )
            elif agent is not None:
                score = 1 / (idx + 1)
            else:
                score = None
            self.add_retrieved_doc(doc, score=score, agent=agent, overwrite=overwrite)

    def add_retrieved_doc(
        self,
        doc: Union[Document, str],
        doc_id: Optional[str] = None,
        score: Optional[float] = None,
        agent: Optional[str] = None,
        overwrite: bool = False,
    ):
        """Add a retrieved document to the query.
        Args:
            doc Document | str: The document (either the object or the text) to add to the query.
            doc_id str: The document ID. If not provided, it will be autogenerated.
            score float: The score of the document when retrieved.
            agent str: The agent that retrieved the document.
        """
        if doc_id is None:
            if isinstance(doc, Document):
                doc_id = doc.did
            else:
                logger.info(f"No doc_id provided. Using '{doc_id}' as doc_id")
                doc_id = f"doc_{len(self.retrieved_docs)+1}"
        if isinstance(doc, str):
            doc = Document(did=doc_id, text=doc)
        doc = self.retrieved_docs.get(doc_id, doc)
        if agent is not None:
            doc.add_retrieved_by(agent, score, overwrite)
        if doc_id in self.retrieved_docs and not overwrite:
            logger.info(f"Query {self.qid} already have a document {doc_id} retrieved.")
            return

        self.retrieved_docs[doc_id] = doc

    def add_agent_answer(
        self, answer: Union[AgentAnswer, str], agent: Optional[str] = None
    ):
        """Add an answer generated by an agent to the query.
        Args:
            answer AgentAnswer | str: The answer (either the object or the text) to add to the query.
            agent str: The id or name of the agent that generated the answer. If not provided, it will be autogenerated.
        """
        if agent is None:
            if isinstance(answer, AgentAnswer):
                agent = answer.agent
            else:
                agent = f"agent_{len(self.answers) + 1}"
                logger.info(f"No agent provided. Using '{agent}' as agent")
        if isinstance(answer, str):
            answer = AgentAnswer(agent=agent, text=answer)
        answer = self.answers.get(agent, answer)
        if answer.agent in self.answers:
            logger.warning(
                f"Answer from agent {answer.agent} already exists in query {self.qid}"
            )
            return
        self.answers[agent] = answer

    def get_qrels(
        self,
        relevance_key: Optional[str] = "relevance",
        relevance_threshold: int = 0,
    ) -> Dict[str, Dict[str, int]]:
        """Get a qrels-formatted dictionary with the relevance of the retrieved documents.
        Args:
            relevance_key str: The key in the answer object that contains an integer with the relevance of the document.
            relevance_threshold int: The minimum relevance value to consider a document as relevant.
                Documents with a relevance lower than this value will be considered as 0.
        """
        qrels = {}
        if len(self.retrieved_docs) == 0:
            logger.warning(
                f"Query {self.qid} does not have any retrieved documents. Returning empty qrels."
            )
        docs_without_relevance = 0
        for did, document in self.retrieved_docs.items():
            if document.evaluation is None:
                docs_without_relevance += 1
                continue
            if isinstance(document.evaluation.answer, int):
                qrels[did] = document.evaluation.answer
            elif isinstance(document.evaluation.answer, dict):
                if (
                    relevance_key is None
                    or relevance_key not in document.evaluation.answer
                ):
                    logger.warning(
                        f"Document {did} does not have a relevance key ({relevance_key})"
                        " in the evaluation. Skipping."
                    )
                    continue
                # check if the relevance is a number or a str that can be converted to an int
                if not isinstance(document.evaluation.answer[relevance_key], int):
                    try:
                        relevance = int(document.evaluation.answer[relevance_key])
                    except ValueError:
                        logger.warning(
                            f"Document {did} has a relevance key ({relevance_key})"
                            f" that cannot be converted to an int ({document.evaluation.answer[relevance_key]})."
                            " Skipping."
                        )
                        continue
                relevance = int(document.evaluation.answer[relevance_key])
                qrels[did] = 0 if relevance < relevance_threshold else relevance
        if docs_without_relevance > 0:
            logger.warning(
                f"Query {self.qid} has {docs_without_relevance} documents without relevance."
            )
        if docs_without_relevance == len(self.retrieved_docs):
            logger.error(f"Query {self.qid} has no documents without relevance.")
        return {self.qid: qrels}

    def get_runs(
        self, agents: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get a trec-style run dictionary with the documents retrieved by the agents provided.
        If no agents are provided, all agents will be included.
        Args:
            agents List[str]: The list of agent IDs to include in the run.
        """
        runs: Dict[str, Dict[str, Dict[str, float]]] = {}
        for did, document in self.retrieved_docs.items():
            for agent, score in document.retrieved_by.items():
                if agents is not None and agent not in agents:
                    continue
                if agent not in runs:
                    runs[agent] = {self.qid: {}}
                runs[agent][self.qid][did] = score

        if len(runs) == 0:
            logger.warning(
                f"Query {self.qid} does not have any retrieved documents with an agent assigned."
            )
        return runs
