"""Base model for dealing with answer evaluators"""

from __future__ import annotations

import asyncio
import itertools
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Type, get_type_hints

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types.configurations import (
    BaseAnswerEvaluatorConfig,
    PairwiseEvaluatorConfig,
)
from ragelo.types.evaluables import AgentAnswer, Document, PairwiseGame
from ragelo.types.queries import Queries
from ragelo.types.query import Query
from ragelo.types.results import AnswerEvaluatorResult
from ragelo.types.types import AnswerEvaluatorTypes


class BaseAnswerEvaluator(BaseEvaluator):
    config: BaseAnswerEvaluatorConfig

    def __init__(
        self,
        config: BaseAnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.config = config
        self.llm_provider = llm_provider

    async def evaluate_async(
        self, eval_sample: tuple[Query, PairwiseGame | AgentAnswer]
    ) -> AnswerEvaluatorResult:
        query, evaluable = eval_sample
        agent: str | tuple[str, str]
        if evaluable.evaluation is not None and not self.config.force:
            return evaluable.evaluation  # type: ignore
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
                response_schema=self.config.llm_response_schema,
            )
        except Exception as e:
            logger.warning(f"Failed to FETCH answers for qid: {query.qid}")
            logger.warning(f"agent(s): {agent}")
            exc = str(e)
            raw_answer = None
            answer = None
        if raw_answer is not None:
            try:
                answer = self._process_answer(raw_answer)
            except ValueError as e:
                logger.warning(
                    f"Failed to PARSE answer for qid: {query.qid} agent(s): {agent}\n"
                    f"Raw answer: {raw_answer}"
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

    def _get_tuples_to_evaluate(
        self, queries: Queries
    ) -> list[tuple[Query, PairwiseGame | AgentAnswer]]:
        tuples_to_eval: list[tuple[Query, PairwiseGame | AgentAnswer]] = []
        all_tuples = 0
        missing_evaluations = 0
        for q in queries:
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
                print(
                    f"All {all_tuples} answers are already evaluated.\n"
                    "If you want to re-evaluate them, use the --force flag"
                )

        return tuples_to_eval

    def evaluate(
        self,
        query: Query | str,
        answer: AgentAnswer | str | None = None,
        answer_a: AgentAnswer | str | None = None,
        answer_b: AgentAnswer | str | None = None,
        retrieved_documents: list[str] | list[Document] | None = None,
        document_metadata: list[dict[str, Any]] | None = None,
        query_metadata: dict[str, Any] | None = None,
        answer_metadata=None,
        answer_a_metadata: dict[str, Any] | None = None,
        answer_b_metadata: dict[str, Any] | None = None,
    ) -> AnswerEvaluatorResult:
        query = self._assemble_query(query, query_metadata)

        if isinstance(retrieved_documents, str):
            retrieved_documents = [retrieved_documents]

        if retrieved_documents:
            retrieved_and_assembled_docs = self._assemble_documents(
                retrieved_documents, document_metadata
            )
            query.retrieved_docs = retrieved_and_assembled_docs

        agent: str | tuple[str, str]

        def run(coroutine):
            return asyncio.run(coroutine)

        if self.config.pairwise:
            if not answer_a or not answer_b:
                raise ValueError("Pairwise evaluations require two answers")
            answer_a = self._assemble_answer(answer_a, answer_a_metadata)
            answer_b = self._assemble_answer(answer_b, answer_b_metadata)
            agent = (answer_a.agent, answer_b.agent)
            evaluable = PairwiseGame(
                agent_a_answer=answer_a,
                agent_b_answer=answer_b,
            )
        else:
            if not answer:
                raise ValueError("Pointwise evaluations require an answer")
            evaluable = self._assemble_answer(answer, answer_metadata)
            agent = evaluable.agent

        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run, self.evaluate_async((query, evaluable)))
                result = future.result()
        except RuntimeError:
            result = asyncio.run(self.evaluate_async((query, evaluable)))

        if result.exception or result.raw_answer is None or result.answer is None:
            raise ValueError(
                f"Failed to evaluate qid: {query.qid} agent(s): {agent}",
                f"Exception: {result.exception}",
            )
        return result

    def batch_evaluate(self, queries: Queries):
        if self.config.pairwise:
            self.__add_pairwise_games(queries)
        super().batch_evaluate(queries)

    def _build_message(
        self, query: Query, answer: AgentAnswer
    ) -> str | list[dict[str, str]]:
        """Builds the message to send to the LLM evaluator"""
        raise NotImplementedError

    def _build_message_pairwise(
        self, query: Query, game: PairwiseGame
    ) -> str | list[dict[str, str]]:
        """Builds the message to send to the LLM evaluator"""
        raise NotImplementedError

    @classmethod
    def from_config(
        cls, config: BaseAnswerEvaluatorConfig, llm_provider: BaseLLMProvider
    ):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> Type[BaseAnswerEvaluatorConfig]:
        return get_type_hints(cls)["config"]

    def __add_pairwise_games(self, queries: Queries):
        if not self.config.pairwise:
            return
        for query in queries:
            query_agents = list(query.answers.keys())
            pairs = list(itertools.combinations(query_agents, 2))
            if self.config.bidirectional:
                pairs += [(b, a) for a, b in pairs]
            random.shuffle(pairs)

            # Filter out games that already exist
            existing_games = {
                (a.agent_a_answer.agent, a.agent_b_answer.agent)
                for a in query.pairwise_games
            }
            games = [g for g in pairs if g not in existing_games]

            games_to_add = self.config.n_games_per_query - len(existing_games)
            games = games[:games_to_add]
            for agent_a, agent_b in games:
                query.pairwise_games.append(
                    PairwiseGame(
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
            raise ValueError(
                "Either the evaluator_name or a config object must be provided"
            )
        evaluator_name = config.evaluator_name
    if isinstance(evaluator_name, str):
        evaluator_name = AnswerEvaluatorTypes(evaluator_name)
    return AnswerEvaluatorFactory.create(
        evaluator_name,
        llm_provider=llm_provider,
        config=config,
        **kwargs,
    )
