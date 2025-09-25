from __future__ import annotations

from typing import Any

from tenacity import RetryError

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
    AnswerEvaluatorTypes,
    BaseAnswerEvaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types.configurations import BaseGroundednessEvaluatorConfig
from ragelo.types.evaluables import AgentAnswerWithDocuments, Document, Evaluable
from ragelo.types.experiment import Experiment
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.query import AgentAnswer, Query
from ragelo.types.results import EvaluatorResult, GroundednessEvaluatorResult
from ragelo.utils import call_async_fn, string_to_template


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.GROUNDEDNESS)
class BaseGroundednessEvaluator(BaseAnswerEvaluator):
    """
    A base class for groundedness evaluators.
    """

    evaluable_name: str = "Groundedness"
    config: BaseGroundednessEvaluatorConfig
    system_prompt = string_to_template(
        """
        You are an expert evaluator tasked with determining whether an AI assistant's response is grounded in the provided documents.
        An answer is considered "grounded" if it accurately references and utilizes information from the provided documents to address the user's query.
        You will be provided with a user's question, the AI assistant's answer, and a set of documents retrieved by a search engine.
        Your task is to assess whether the answer is grounded in the provided documents.
        When available, answers will cite specific documents by placing their IDs into square brackets.
        
        Each document is scored in a scale of 0 to 2, where:
            - 0: the answer is not grounded in the provided documents
            - 1: the answer is somewhat grounded in the provided documents
            - 2: the answer is highly grounded in the provided documents"""
    )
    user_prompt = string_to_template(
        """
        {%- if query %}
        The user's question is:
        {{ query }}
        {%- endif %}
        {%- if retrieved_docs %}
        The following documents were retrieved by a search engine in response to the user's question:
        {{ retrieved_docs }}
        {%- endif %}
        {%- if answer %}
        The AI assistant's answer is:
        {{ answer }}
        {%- endif %}"""
    )

    def __init__(self, config: BaseGroundednessEvaluatorConfig, llm_provider: BaseLLMProvider):
        super().__init__(config, llm_provider)

    def evaluate(
        self,
        query: Query | str,
        retrieved_docs: list[Document | str],
        answer: AgentAnswer | str,
        document_metadata: list[dict[str, Any]] | None = None,
        query_metadata: dict[str, Any] | None = None,
        answer_metadata: dict[str, Any] | None = None,
    ) -> GroundednessEvaluatorResult:
        """Evaluates a single query-document pair. Returns the raw answer and the processed answer.
        Args:
            query (Query | str): The query to evaluate.
                If a string is provided, a Query object will be created with the provided query_metadata.
            document (Document | str): The document to evaluate.
                If a string is provided, a Document object will be created with the provided doc_metadata.
            query_metadata (dict[str, Any] | None): The metadata for the query.
            doc_metadata (dict[str, Any] | None): The metadata for the document.
        """
        query = Query.assemble_query(query, query_metadata)
        docs = Document.assemble_documents(retrieved_docs, query.qid, document_metadata)
        answer = AgentAnswer.assemble_answer(answer, query.qid, metadata=answer_metadata)

        result = call_async_fn(self.evaluate_async, (query, docs, answer))

        if result.exception or result.raw_answer is None or result.answer is None:
            raise ValueError(
                f"Failed to evaluate groundedness for qid: {query.qid}",
                f"Exception: {result.exception}",
            )
        return result

    def _build_message(self, query: Query, answer: AgentAnswerWithDocuments) -> LLMInputPrompt:
        """Builds the message to send to the LLM evaluator"""

        context = {
            "query": query.query,
            "answer": answer.agent_answer.text,
            "retrieved_docs": "\n".join([f"[{doc.did}]: {doc.text}" for doc in answer.documents.values()]),
        }
        user_message = self.user_prompt.render(**context)
        system_message = self.system_prompt.render() if self.system_prompt else None
        return LLMInputPrompt(
            system_prompt=system_message,
            user_message=user_message,
        )

    async def evaluate_async(self, eval_sample: tuple[Query, AgentAnswerWithDocuments]) -> GroundednessEvaluatorResult:
        """
        Evaluates a single sample (either an answer or a pairwise game) asynchronously.
        Args:
            eval_sample (tuple[Query, Evaluable]): The query and evaluable to evaluate.
        """

        (query, answer_with_docs) = eval_sample

        if answer_with_docs.agent_answer.groundedness_evaluation is not None and not self.config.force:
            return GroundednessEvaluatorResult(
                qid=query.qid,
                agent=answer_with_docs.agent_answer.agent,
                raw_answer=answer_with_docs.agent_answer.groundedness_evaluation.raw_answer,
                answer=answer_with_docs.agent_answer.groundedness_evaluation.answer,
            )

        input_prompt = self._build_message(query, answer_with_docs)

        exc = None
        try:
            llm_response = await self.llm_provider.call_async(
                input=input_prompt,
                response_schema=self.config.llm_response_schema,
            )
            llm_response = self._process_answer(llm_response)
        except ValueError as e:
            logger.warning(f"Failed to PARSE answer for qid: {query.qid}\nRaw answer: {llm_response.raw_answer}")
            exc = str(e)
        except Exception as e:
            logger.warning(f"Failed to FETCH answers for qid: {query.qid}")
            if isinstance(e, RetryError):
                exc = str(e.last_attempt.exception())
            else:
                exc = str(e)

        return GroundednessEvaluatorResult(
            qid=query.qid,
            agent=answer_with_docs.agent_answer.agent,
            raw_answer=llm_response.raw_answer,
            answer=llm_response.parsed_answer,
            exception=exc,
        )

    def _get_tuples_to_evaluate(self, experiment: Experiment) -> list[tuple[Query, AgentAnswerWithDocuments]]:
        """
        Creates the list of pairs (query, evaluable) to evaluate
        """
        tuples_to_eval: list[tuple[Query, AgentAnswerWithDocuments]] = []
        all_tuples = 0
        missing_evaluations = 0
        for q in experiment:
            retrieved_docs = q.retrieved_docs
            for a in q.answers.values():
                all_tuples += 1

                tuples_to_eval.append(
                    (
                        q,
                        AgentAnswerWithDocuments(qid=q.qid, agent_answer=a, documents=retrieved_docs),
                    )
                )
                if a.groundedness_evaluation is None:
                    missing_evaluations += 1

        if missing_evaluations == 0 and not self.config.force:
            logger.info(
                f"All {all_tuples} answers are already evaluated for groundedness.\n"
                "If you want to re-evaluate them, use the --force flag"
            )

        return tuples_to_eval
