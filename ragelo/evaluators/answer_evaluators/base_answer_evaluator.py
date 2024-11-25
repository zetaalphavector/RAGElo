"""Base model for dealing with answer evaluators"""

from __future__ import annotations

import asyncio
import itertools
import random
from typing import Any, Callable, Sequence, Type, get_type_hints

from tenacity import RetryError

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types.configurations import (
    BaseAnswerEvaluatorConfig,
    PairwiseEvaluatorConfig,
)
from ragelo.types.evaluables import AgentAnswer, Document, Evaluable, PairwiseGame
from ragelo.types.experiment import Experiment
from ragelo.types.query import Query
from ragelo.types.results import AnswerEvaluatorResult
from ragelo.types.types import AnswerEvaluatorTypes
from ragelo.utils import call_async_fn


class BaseAnswerEvaluator(BaseEvaluator):
    config: BaseAnswerEvaluatorConfig
    evaluable_name: str = "Agent Answer"

    def __init__(
        self,
        config: BaseAnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.config = config
        self.llm_provider = llm_provider

    def evaluate(
        self,
        query: Query | str,
        answer: AgentAnswer | str | None = None,
        answer_a: AgentAnswer | str | None = None,
        answer_b: AgentAnswer | str | None = None,
        retrieved_documents: list[Document | str] | str | None = None,
        document_metadata: list[dict[str, Any]] | None = None,
        query_metadata: dict[str, Any] | None = None,
        answer_metadata: dict[str, Any] | None = None,
        answer_a_metadata: dict[str, Any] | None = None,
        answer_b_metadata: dict[str, Any] | None = None,
    ) -> AnswerEvaluatorResult:
        """
        Evaluates a single sample using an answer evaluator.
        Args:
            query(Query|str): The query to evaluate.
                If a string is provided, a Query object will be created with the provided query_metadata.
            answer(Optional[AgentAnswer|str]): The answer generated by an agent to evaluate.
                If a string is provided, the AgentAnswer object will be created with the provided answer_metadata
            answer_a(Optional[AgentAnswer|str]): The first answer generated by an agent to evaluate.
                Only used if the evaluator is pairwise.
            answer_b(Optional[AgentAnswer|str]): The second answer generated by an agent to evaluate.
                Only used if the evaluator is pairwise.
            retrieved_documents(Optional[list[str]|list[Document]]): The documents retrieved by the agent.
            document_metadata(Optional[list[dict[str, Any]]): The metadata of the documents retrieved by the agent.
            query_metadata(Optional[dict[str, Any]]): The metadata of the query.
            answer_metadata(Optional[dict[str, Any]]): The metadata of the answer.
            answer_a_metadata(Optional[dict[str, Any]]): The metadata of the first answer.
                Only used if the evaluator is pairwise.
            answer_b_metadata(Optional[dict[str, Any]]): The metadata of the second answer.
                Only used if the evaluator is pairwise.
        Returns:
            AnswerEvaluatorResult: The result of the evaluation.
        """
        query = Query.assemble_query(query, query_metadata)

        if isinstance(retrieved_documents, str):
            retrieved_documents = [retrieved_documents]

        if retrieved_documents:
            retrieved_and_assembled_docs = Document.assemble_documents(
                retrieved_documents, query.qid, document_metadata
            )
            query.retrieved_docs = retrieved_and_assembled_docs

        agent: str | tuple[str, str]
        evaluable: AgentAnswer | PairwiseGame

        def run(coroutine):
            return asyncio.run(coroutine)

        if self.config.pairwise:
            if not answer_a or not answer_b:
                raise ValueError("Pairwise evaluations require two answers")
            answer_a = AgentAnswer.assemble_answer(answer_a, query.qid, metadata=answer_a_metadata)
            answer_b = AgentAnswer.assemble_answer(answer_b, query.qid, metadata=answer_b_metadata)
            agent = (answer_a.agent, answer_b.agent)
            evaluable = PairwiseGame(
                qid=query.qid,
                agent_a_answer=answer_a,
                agent_b_answer=answer_b,
            )
        else:
            if not answer:
                raise ValueError("Pointwise evaluations require an answer")
            evaluable = AgentAnswer.assemble_answer(answer, query.qid, metadata=answer_metadata)
            agent = evaluable.agent
        result = call_async_fn(self.evaluate_async, (query, evaluable))

        if result.exception or result.raw_answer is None or result.answer is None:
            raise ValueError(
                f"Failed to evaluate qid: {query.qid} agent(s): {agent}",
                f"Exception: {result.exception}",
            )
        return result

    def evaluate_experiment(self, experiment: Experiment, n_threads: int | None = None):
        """
        Trigger the evaluation of all the queries in the experiment.
        The evaluation is done in asynchronously with the number of threads defined in config.n_processes parameter.
        This can be overwritten by the n_threads parameter.

        Args:
            experiment(Experiment): The experiment to evaluate.
            n_threads(int): The number of threads to use for the evaluation.
                If None, the number of threads defined in the config will be used.
        """
        if self.config.pairwise:
            self.__add_pairwise_games(experiment)
        super().evaluate_experiment(experiment, n_threads)

    async def evaluate_async(self, eval_sample: tuple[Query, Evaluable]) -> AnswerEvaluatorResult:
        """
        Evaluates a single sample asynchronously
        """
        agent: str | tuple[str, str]
        query, evaluable = eval_sample

        if not isinstance(evaluable, AgentAnswer) or not isinstance(evaluable, PairwiseGame):
            type_name = type(evaluable).__name__
            raise ValueError(f"can't evaluate a {type_name} in a Retrieval Evaluator")

        if evaluable.evaluation is not None and not self.config.force:
            if isinstance(evaluable, AgentAnswer):
                return AnswerEvaluatorResult(
                    qid=query.qid,
                    agent=evaluable.agent,
                    raw_answer=evaluable.evaluation.raw_answer,
                    answer=evaluable.evaluation.answer,
                    pairwise=False,
                )
            return AnswerEvaluatorResult(
                qid=query.qid,
                agent_a=evaluable.agent_a_answer.agent,
                agent_b=evaluable.agent_b_answer.agent,
                raw_answer=evaluable.evaluation.raw_answer,
                answer=evaluable.evaluation.answer,
                pairwise=True,
            )

        if isinstance(evaluable, AgentAnswer):
            agent = evaluable.agent
            prompt = self._build_message(query, evaluable)
        elif isinstance(evaluable, PairwiseGame):
            agent = (evaluable.agent_a_answer.agent, evaluable.agent_b_answer.agent)
            prompt = self._build_message_pairwise(query, evaluable)
        else:
            raise ValueError(f"Unknown evaluable type {type(evaluable)}")

        exc = None
        try:
            raw_answer = await self.llm_provider.call_async(
                prompt,
                answer_format=self.config.llm_answer_format,
                response_format=self.config.llm_response_schema,
            )
        except Exception as e:
            logger.warning(f"Failed to FETCH answers for qid: {query.qid}")
            logger.warning(f"agent(s): {agent}")
            if isinstance(e, RetryError):
                exc = str(e.last_attempt.exception())
            else:
                exc = str(e)
            raw_answer = None
            answer = None
        if raw_answer is not None:
            try:
                answer = self._process_answer(raw_answer)
            except ValueError as e:
                logger.warning(
                    f"Failed to PARSE answer for qid: {query.qid} agent(s): {agent}\n" f"Raw answer: {raw_answer}"
                )
                exc = str(e)
                answer = None
        if isinstance(evaluable, AgentAnswer):
            return AnswerEvaluatorResult(
                qid=query.qid,
                agent=evaluable.agent,
                raw_answer=raw_answer,
                answer=answer,
                pairwise=False,
                exception=exc,
            )
        return AnswerEvaluatorResult(
            qid=query.qid,
            agent_a=evaluable.agent_a_answer.agent,
            agent_b=evaluable.agent_b_answer.agent,
            raw_answer=raw_answer,
            answer=answer,
            pairwise=True,
            exception=exc,
        )

    def _get_tuples_to_evaluate(self, experiment: Experiment) -> Sequence[tuple[Query, Evaluable]]:
        """
        Creates the list of pairs (query, evaluable) to evaluate
        """
        tuples_to_eval: list[tuple[Query, PairwiseGame | AgentAnswer]] = []
        all_tuples = 0
        missing_evaluations = 0
        for q in experiment:
            if self.config.pairwise:
                for g in q.pairwise_games:
                    all_tuples += 1
                    tuples_to_eval.append((q, g))
                    if g.evaluation is None:
                        missing_evaluations += 1

            else:
                for a in q.answers.values():
                    all_tuples += 1
                    tuples_to_eval.append((q, a))
                    if a.evaluation is None:
                        missing_evaluations += 1

        if missing_evaluations == 0:
            logger.info("All answers have been evaluated")
            if self.config.verbose and not self.config.force:
                logger.warning(
                    f"All {all_tuples} answers are already evaluated.\n"
                    "If you want to re-evaluate them, use the --force flag"
                )

        return tuples_to_eval

    def _build_message(self, query: Query, answer: AgentAnswer) -> str | list[dict[str, str]]:
        """Builds the message to send to the LLM evaluator"""
        raise NotImplementedError

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> str | list[dict[str, str]]:
        """Builds the message to send to the LLM evaluator"""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: BaseAnswerEvaluatorConfig, llm_provider: BaseLLMProvider):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> Type[BaseAnswerEvaluatorConfig]:
        return get_type_hints(cls)["config"]

    def __add_pairwise_games(self, experiment: Experiment):
        if not self.config.pairwise:
            return
        if not isinstance(self.config, PairwiseEvaluatorConfig):
            raise ValueError("Trying to add pairwise games to a non-pairwise evaluator")
        for query in experiment:
            query_agents = list(query.answers.keys())
            pairs = list(itertools.combinations(query_agents, 2))
            if self.config.bidirectional:
                pairs += [(b, a) for a, b in pairs]
            random.shuffle(pairs)

            # Filter out games that already exist
            existing_games = {(a.agent_a_answer.agent, a.agent_b_answer.agent) for a in query.pairwise_games}
            games = [g for g in pairs if g not in existing_games]

            games_to_add = self.config.n_games_per_query - len(existing_games)
            games = games[:games_to_add]
            for agent_a, agent_b in games:
                query.pairwise_games.append(
                    PairwiseGame(
                        qid=query.qid,
                        agent_a_answer=query.answers[agent_a],
                        agent_b_answer=query.answers[agent_b],
                    )
                )

    def _prepare_documents(self, query: Query) -> str:
        documents = []
        for did, d in query.retrieved_docs.items():
            if self.config.document_relevance_threshold is not None:
                # Skip documents with relevance below the threshold
                if d.evaluation is None:
                    continue
                # check if evaluation.answer is an integer or a string that can be converted to an integer
                score = d.evaluation.answer
                if isinstance(score, str) and score.isdigit():
                    score = int(score)
                if not isinstance(score, int):
                    continue
                if score < self.config.document_relevance_threshold:
                    continue
            if self.config.document_filter is not None:
                if d.evaluation is None:
                    continue
                if not self.config.document_filter(str(d.evaluation.answer)):
                    continue
            formatters = {
                "did": did,
                "doc": d.text,
                "annotation": d.evaluation.answer if d.evaluation else None,
            }
            documents.append(self.config.document_template.format(**formatters))
        if len(documents) == 0:
            return "NO DOCUMENTS WERE RETRIEVED"
        return "\n".join(documents)


class AnswerEvaluatorFactory:
    registry: dict[AnswerEvaluatorTypes, Type[BaseAnswerEvaluator]] = {}

    @classmethod
    def register(cls, name: AnswerEvaluatorTypes) -> Callable:
        def inner_wrapper(wrapped_class: Type[BaseAnswerEvaluator]):
            if name in cls.registry:
                logger.warning(f"Overwriting {name} in registry")
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        evaluator_name: AnswerEvaluatorTypes,
        llm_provider: BaseLLMProvider | str,
        config: BaseAnswerEvaluatorConfig | None = None,
        **kwargs,
    ) -> BaseAnswerEvaluator:
        if evaluator_name not in cls.registry:
            raise ValueError(f"Unknown evaluator {evaluator_name}")
        if isinstance(llm_provider, str):
            llm_provider_instance = get_llm_provider(llm_provider, **kwargs)
        else:
            llm_provider_instance = llm_provider
        if config is None:
            class_ = cls.registry[evaluator_name]
            type_config = class_.get_config_class()
            valid_keys = [field for field in type_config.get_model_fields()]
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            config = type_config(**valid_args)
        return cls.registry[evaluator_name].from_config(config, llm_provider_instance)


def get_answer_evaluator(
    evaluator_name: AnswerEvaluatorTypes | str | None = None,
    llm_provider: BaseLLMProvider | str = "openai",
    config: BaseAnswerEvaluatorConfig | None = None,
    **kwargs,
) -> BaseAnswerEvaluator:
    if evaluator_name is None:
        # get the name from the config
        if config is None:
            raise ValueError("Either the evaluator_name or a config object must be provided")
        evaluator_name = config.evaluator_name
    if isinstance(evaluator_name, str):
        evaluator_name = AnswerEvaluatorTypes(evaluator_name)
    return AnswerEvaluatorFactory.create(
        evaluator_name,
        llm_provider=llm_provider,
        config=config,
        **kwargs,
    )
