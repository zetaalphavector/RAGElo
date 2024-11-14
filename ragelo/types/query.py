from __future__ import annotations

from typing import Any

from ragelo.logger import logger
from ragelo.types.evaluables import AgentAnswer, Document, PairwiseGame
from ragelo.types.pydantic_models import BaseModel, validator
from ragelo.types.results import AnswerEvaluatorResult, RetrievalEvaluatorResult


class Query(BaseModel):
    """A user query that can have retrieved documents and agent answers.
    Args:
        qid str: The query ID.
        query str: The query text.
        metadata Optional[dict[str, Any]]: Metadata that can be templated in the prompt.
        retrieved_docs dict[str, Document]: A dictionary of retrieved documents, where the key is the document ID.
        answers list[AgentAnswer]: The list of agent answers.
        pairwise_games list[PairwiseGame]: The list games to be played between agent's answers. Generated by the evaluator.
    """

    qid: str
    query: str
    metadata: dict[str, Any] | None = None
    retrieved_docs: dict[str, Document] = {}
    answers: dict[str, AgentAnswer] = {}
    pairwise_games: list[PairwiseGame] = []

    def add_metadata(self, metadata: dict[str, Any] | None):
        """Adds metadata to the query that may be templated in the prompt.
        Args:
            metadata dict[str, Any]: The metadata to add to the query.
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
        docs: list,
        agent: str | None = None,
        force: bool = False,
    ):
        """Add a list of retrieved documents to the query.
        Args:
            docs list[Document | str] | list[Tuple[Document | str], float]: The list of documents (either the objects or the texts) to add to the query. Optionally, a score can be provided for each document.
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
            self.add_retrieved_doc(doc, score=score, agent=agent, force=force)

    def add_retrieved_doc(
        self,
        doc: Document,
        score: float | None = None,
        agent: str | None = None,
        force: bool = False,
    ):
        """Add a retrieved document to the query.
        Args:
            doc Document: The document object retrieved for the query.
            score float: The score of the document when retrieved.
            agent str: The agent that retrieved the document.
        """
        doc = self.retrieved_docs.get(doc.did, doc.model_copy(deep=True))
        if agent is not None:
            doc.add_retrieved_by(agent, score, force)
        if doc.did in self.retrieved_docs and not force:
            logger.info(
                f"Query {self.qid} already have a document {doc.did} retrieved."
            )
            return

        self.retrieved_docs[doc.did] = doc

    def add_agent_answer(self, answer: AgentAnswer, force: bool = False):
        """Add an answer generated by an agent to the query.
        Args:
            answer AgentAnswer | str: The answer (either the object or the text) to add to the query.
            agent str: The id or name of the agent that generated the answer. If not provided, it will be autogenerated.
        """
        agent = answer.agent
        answer = self.answers.get(agent, answer)
        if answer.agent in self.answers and not force:
            logger.warning(
                f"Answer from agent {answer.agent} already exists in query {self.qid}"
            )
            return
        if answer.agent in self.answers:
            logger.info(
                f"Answer from agent {answer.agent} already exists in query {self.qid}. Overwriting."
            )
        self.answers[agent] = answer

    def add_evaluation(
        self,
        evaluation: RetrievalEvaluatorResult | AnswerEvaluatorResult,
        force: bool = False,
    ):
        if isinstance(evaluation, RetrievalEvaluatorResult):
            did = evaluation.did
            if did not in self.retrieved_docs:
                logger.warning(
                    f"Trying to add evaluation for non-retrieved document {did} in query {self.qid}"
                )
            if self.retrieved_docs[did].evaluation is not None and not force:
                logger.warning(
                    f"Document {did} in query {self.qid} already has an evaluation."
                )
                return
            if self.retrieved_docs[did].evaluation is not None:
                logger.info(
                    f"Document {did} in query {self.qid} already has an evaluation. Overwriting."
                )
            self.retrieved_docs[did].evaluation = evaluation
            return
        if evaluation.pairwise:
            agent_a = evaluation.agent_a
            agent_b = evaluation.agent_b
            if agent_a not in self.answers:
                logger.warning(
                    "Trying to add a pairwise evaluation for a comparison between agents "
                    f"{agent_a} and {agent_b}, but {agent_a} does not have an answer for query {self.qid}"
                )
            if agent_b not in self.answers:
                logger.warning(
                    "Trying to add a pairwise evaluation for a comparison between agents "
                    f"{agent_a} and {agent_b}, but {agent_b} does not have an answer for query {self.qid}"
                )
            for game in self.pairwise_games:
                if (
                    game.agent_a_answer.agent == agent_a
                    and game.agent_b_answer.agent == agent_b
                ):
                    if game.evaluation is not None and not force:
                        logger.warning(
                            f"Query {self.qid} already has an evaluation for agents {agent_a} and {agent_b}."
                        )
                        return
                    if game.evaluation is not None:
                        logger.info(
                            f"Query {self.qid} already has an evaluation for agents {agent_a} and {agent_b}. Overwriting."
                        )
                    game.evaluation = evaluation
                    return
            logger.warning(
                f"Trying to add a pairwise evaluation for a comparison between agents {agent_a} and {agent_b},"
                f" but no game was found for query {self.qid}"
            )
            return
        if evaluation.agent is None:
            raise ValueError(
                "A pointwise AnswerEvaluatorResult must have an agent assigned to it."
            )
        agent = evaluation.agent
        if agent not in self.answers:
            logger.warning(
                f"Trying to add evaluation for agent {agent} in query {self.qid}, but {agent} does not have an answer."
            )
        if self.answers[agent].evaluation is not None and not force:
            logger.warning(
                f"Agent {agent} in query {self.qid} already has an evaluation."
            )
            return
        if self.answers[agent].evaluation is not None:
            logger.info(
                f"Agent {agent} in query {self.qid} already has an evaluation. Overwriting."
            )
        self.answers[agent].evaluation = evaluation

    def get_qrels(
        self,
        relevance_key: str | None = "relevance",
        relevance_threshold: int = 0,
    ) -> dict[str, int]:
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
                relevance = document.evaluation.answer
            elif isinstance(document.evaluation.answer, str):
                try:
                    relevance = int(document.evaluation.answer)
                except ValueError:
                    logger.warning(
                        f"Document {did} has a relevance key ({relevance_key})"
                        f" that cannot be converted to an int ({document.evaluation.answer})."
                        " Skipping."
                    )
                    continue
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
        return qrels

    def get_runs(
        self, agents: list[str] | None = None
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Get a trec-style run dictionary with the documents retrieved by the agents provided.
        If no agents are provided, all agents will be included.
        Args:
            agents list[str]: The list of agent IDs to include in the run.
        """
        runs: dict[str, dict[str, dict[str, float]]] = {}
        for did, document in self.retrieved_docs.items():
            for agent, score in document.retrieved_by.items():
                if agents is not None and agent not in agents:
                    continue
                if agent not in runs:
                    runs[agent] = {self.qid: {}}
                runs[agent][self.qid][did] = float(score)

        if len(runs) == 0:
            logger.warning(
                f"Query {self.qid} does not have any retrieved documents with an agent assigned."
            )
        return runs

    @classmethod
    def assemble_query(
        cls, query: Query | str, metadata: dict[str, Any] | None = None
    ) -> Query:
        """Assembles a Query object from a Query object or a query text.
        Args:
            query Query | str: The query object or the query text.
            metadata dict[str, Any]: Metadata to add to the query.
        """
        if isinstance(query, Query):
            query.add_metadata(metadata)
            return query
        return cls(qid="<no_qid>", query=query, metadata=metadata)