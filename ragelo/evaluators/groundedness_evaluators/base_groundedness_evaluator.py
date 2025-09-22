from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple, Type, get_type_hints

import rich
from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types.answer_formats import GroundednessEvaluatorFormat
from ragelo.types.configurations import BaseGroundednessEvaluatorConfig
from ragelo.types.evaluables import AgentAnswerWithDocuments, Document, Evaluable
from ragelo.types.experiment import Experiment
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.query import AgentAnswer, Query
from ragelo.types.results import EvaluatorResult, GroundednessEvaluatorResult
from ragelo.utils import call_async_fn, string_to_template
from tenacity import RetryError


class BaseGroundednessEvaluator(BaseEvaluator):
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

    def __init__(
        self, config: BaseGroundednessEvaluatorConfig, llm_provider: BaseLLMProvider
    ):
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
        answer = AgentAnswer.assemble_answer(
            answer, query.qid, metadata=answer_metadata
        )

        result = call_async_fn(self.evaluate_async, (query, docs, answer))

        if result.exception or result.raw_answer is None or result.answer is None:
            raise ValueError(
                f"Failed to evaluate groundedness for qid: {query.qid}",
                f"Exception: {result.exception}",
            )
        return result

    def _build_message(
        self, query: Query, answer: AgentAnswerWithDocuments
    ) -> LLMInputPrompt:
        """Builds the message to send to the LLM evaluator"""

        context = {
            "query": query.query,
            "answer": answer.agent_answer.text,
            "retrieved_docs": "\n".join(
                [f"[{doc.did}]: {doc.text}" for doc in answer.documents.values()]
            ),
        }
        user_message = self.user_prompt.render(**context)
        system_message = self.system_prompt.render() if self.system_prompt else None
        return LLMInputPrompt(
            system_prompt=system_message,
            user_message=user_message,
        )

    async def evaluate_async(
        self, eval_sample: tuple[Query, AgentAnswerWithDocuments]
    ) -> GroundednessEvaluatorResult:
        """
        Evaluates a single sample (either an answer or a pairwise game) asynchronously.
        Args:
            eval_sample (tuple[Query, Evaluable]): The query and evaluable to evaluate.
        """

        (query, answer_with_docs) = eval_sample

        if (
            answer_with_docs.agent_answer.groundedness_evaluation is not None
            and not self.config.force
        ):
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
            logger.warning(
                f"Failed to PARSE answer for qid: {query.qid}\nRaw answer: {llm_response.raw_answer}"
            )
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

    def _get_tuples_to_evaluate(
        self, experiment: Experiment
    ) -> list[tuple[Query, AgentAnswerWithDocuments]]:
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
                        AgentAnswerWithDocuments(
                            qid=q.qid, agent_answer=a, documents=retrieved_docs
                        ),
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

    def _aggregate_groundedness_scores(
        self, experiment: Experiment
    ) -> list[Tuple[str, float]]:
        """
        Aggregates the groundedness scores for each query in the experiment.
        Args:
            experiment (Experiment): The experiment to aggregate the scores for.
        Returns:
            dict[str, float]: A dictionary mapping each query ID to its average groundedness score.
        """
        groundedness_scores = {}
        total_scores_per_agent = {}
        counts_per_agent = {}
        for query in experiment:

            for answer in query.answers.values():
                if answer.groundedness_evaluation is not None:
                    eval_answer = answer.groundedness_evaluation.answer
                    if isinstance(
                        answer.groundedness_evaluation.answer,
                        GroundednessEvaluatorFormat,
                    ):
                        eval_answer = vars(answer.groundedness_evaluation.answer)
                    if isinstance(eval_answer, dict):
                        total_scores_per_agent[answer.agent] = (
                            total_scores_per_agent.get(answer.agent, 0)
                            + (float(eval_answer["groundedness_score"]) / 2)
                        )
                        counts_per_agent[answer.agent] = (
                            counts_per_agent.get(answer.agent, 0) + 1
                        )

        for agent in total_scores_per_agent:
            groundedness_scores[agent] = (
                total_scores_per_agent[agent] / counts_per_agent[agent]
                if counts_per_agent[agent] > 0
                else 0.0
            )

        sorted_groundedness_scores = sorted(
            groundedness_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_groundedness_scores

    def evaluate_experiment(self, experiment: Experiment, n_threads: int | None = None):
        """
        Trigger the evaluator for all the supported evaluables in the experiment.
        The evaluation is done in asynchronously with the number of threads defined in config.n_processes parameter.
        This can be overwritten by the n_threads parameter.

        Args:
            experiment(Experiment): The experiment to evaluate.
            n_threads(int): The number of threads to use for the evaluation.
                If None, the number of threads defined in the config will be used.
        """
        n_threads = n_threads or self.config.n_processes
        call_async_fn(self._evaluate_experiment_async, experiment, n_threads)

        aggregate_groundedness_scores = self._aggregate_groundedness_scores(experiment)

        if experiment.rich_print:
            rich.print(
                "-------[bold white] Groundedness scores [0-1] [/bold white]-------"
            )
        else:
            print("\n------- Groundedness scores [0-1] -------")
        for agent, score in aggregate_groundedness_scores:
            if experiment.rich_print:
                rich.print(f"[bold white]{agent:<18}[/bold white]: {score:.2f}")
            else:
                print(f"{agent:<18}: {score:.1f}")
