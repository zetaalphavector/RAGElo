"""Base model for dealing with answer evaluators"""

import asyncio
import itertools
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Type, get_type_hints

from tqdm import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types import (
    AgentAnswer,
    AnswerEvaluatorResult,
    AnswerEvaluatorTypes,
    AnswerFormat,
    BaseAnswerEvaluatorConfig,
    Document,
    PairwiseEvaluatorConfig,
    PairwiseGame,
    Query,
)


class BaseAnswerEvaluator(BaseEvaluator):
    config: BaseAnswerEvaluatorConfig

    def __init__(
        self,
        config: BaseAnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.config = config
        self.llm_provider = llm_provider
        self.output_columns = config.output_columns
        if config.answer_format == AnswerFormat.MULTI_FIELD_JSON:
            self.config.scoring_keys = config.scoring_keys
            self.output_columns = [
                "qid",
                "agent",
                "raw_answer",
            ] + self.config.scoring_keys
        elif config.answer_format == AnswerFormat.JSON:
            config.scoring_keys = []
        elif config.scoring_key and config.scoring_key not in self.output_columns:
            print(f"Adding scoring key {config.scoring_key} to output columns")
            self.output_columns.append(self.config.scoring_key)
        if config.scoring_keys:
            missing_keys = [
                key for key in config.scoring_keys if key not in self.output_columns
            ]
            self.output_columns.extend(missing_keys)

    async def _async_batch_evaluate(self, queries: list[Query]) -> list[Query]:
        use_progress_bar = self.config.use_progress_bar
        failed_queries = 0
        queries = self.__prepare_queries(queries)
        tuples_to_eval = self.__get_tuples_to_evaluate(queries)

        if len(tuples_to_eval) == 0:
            return queries
        pbar = tqdm(
            total=len(tuples_to_eval),
            ncols=100,
            desc="Evaluating answers",
            disable=not use_progress_bar,
            leave=False,
            position=0,
        )

        awaitables_ended = False
        pending: set[asyncio.Future] = set()
        aws = map(self._async_evaluate, tuples_to_eval)
        aws = iter(aws)
        evaluations = []
        # while there are pending tasks or not all tasks are done
        while pending or not awaitables_ended:
            # while there are less than n_processes pending tasks
            while len(pending) < self.config.n_processes and not awaitables_ended:
                try:
                    aw = next(aws)
                except StopIteration:
                    awaitables_ended = True  # all tasks have been scheduled
                else:
                    pending.add(asyncio.ensure_future(aw))
            if not pending:
                break

            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            while done:
                evaluation = await done.pop()
                pbar.update()
                if evaluation.exception:
                    failed_queries += 1
                    continue
                evaluations.append(evaluation)
        pbar.close()
        self._add_answers_evaluations(queries, evaluations)
        if self.config.verbose:
            print("✅ Done!")
            print("Failed evaluations:", failed_queries)
            print(f"Total evaluations: {len(evaluations)}")
        return queries

    def evaluate(
        self,
        query: Query | str,
        answer: Optional[AgentAnswer | str] = None,
        answer_a: Optional[AgentAnswer | str] = None,
        answer_b: Optional[AgentAnswer | str] = None,
        retrieved_documents: Optional[list[str] | list[Document]] = None,
        document_metadata: Optional[list[dict[str, Any]]] = None,
        query_metadata: Optional[dict[str, Any]] = None,
        answer_metadata=None,
        answer_a_metadata: Optional[dict[str, Any]] = None,
        answer_b_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Any]:
        query = self._assemble_query(query, query_metadata)
        if isinstance(retrieved_documents, str):
            retrieved_documents = [retrieved_documents]
        if retrieved_documents:
            retrieved_and_assembled_docs = self._assemble_documents(
                retrieved_documents, document_metadata
            )
            query.retrieved_docs = retrieved_and_assembled_docs
        agent: str | tuple[str, str]
        if self.config.pairwise:
            if not answer_a or not answer_b:
                raise ValueError("Pairwise evaluations require two answers")
            answer_a = self._assemble_answer(answer_a, answer_a_metadata)
            answer_b = self._assemble_answer(answer_b, answer_b_metadata)
            agent = (answer_a.agent, answer_b.agent)
            game = PairwiseGame(
                agent_a_answer=answer_a,
                agent_b_answer=answer_b,
            )
            try:
                asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run, self._async_evaluate((query, game))
                    )
                    result = future.result()
            except RuntimeError:
                result = asyncio.run(self._async_evaluate((query, game)))
        else:
            if not answer:
                raise ValueError("Pointwise evaluations require an answer")
            answer = self._assemble_answer(answer, answer_metadata)
            agent = answer.agent
            try:
                asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run, self._async_evaluate((query, answer))
                    )
                    result = future.result()
            except RuntimeError:
                result = asyncio.run(self._async_evaluate((query, answer)))
        if result.exception or result.raw_answer is None or result.answer is None:
            raise ValueError(
                f"Failed to evaluate qid: {query.qid} agent(s): {agent}",
                f"Exception: {result.exception}",
            )
        return result.raw_answer, result.answer

    async def _async_evaluate(
        self, eval_sample: tuple[Query, PairwiseGame | AgentAnswer]
    ) -> AnswerEvaluatorResult:
        query, evaluable = eval_sample
        agent: str | tuple[str, str]
        if isinstance(evaluable, AgentAnswer):
            agent = evaluable.agent
        elif isinstance(evaluable, PairwiseGame):
            agent = (evaluable.agent_a_answer.agent, evaluable.agent_b_answer.agent)
        else:
            raise ValueError(f"Unknown evaluable type {type(evaluable)}")

        exc = None
        if isinstance(evaluable, AgentAnswer):
            prompt = self._build_message(query, evaluable)
        else:
            prompt = self._build_message_pairwise(query, evaluable)
        try:
            raw_answer = await self.llm_provider.call_async(prompt)
        except Exception as e:
            logger.warning(f"Failed to FETCH answers for qid: {query.qid}")
            logger.warning(f"agent(s): {agent}")
            exc = str(e)
            raw_answer = None
        try:
            answer = self._process_answer(raw_answer) if raw_answer else None
        except ValueError as e:
            logger.warning(
                f"Failed to PARSE answer for qid: {query.qid} agent(s): {agent}\n"
                f"Raw answer: {raw_answer}"
            )
            exc = str(e)
            answer = None
        if isinstance(evaluable, AgentAnswer):
            output_file = self.config.answers_evaluations_path
            ans = AnswerEvaluatorResult(
                qid=query.qid,
                agent=evaluable.agent,
                raw_answer=raw_answer,
                answer=answer,
                pairwise=False,
                exception=exc,
            )
        else:
            output_file = self.config.games_evaluations_path
            ans = AnswerEvaluatorResult(
                qid=query.qid,
                agent_a=evaluable.agent_a_answer.agent,
                agent_b=evaluable.agent_b_answer.agent,
                raw_answer=raw_answer,
                answer=answer,
                pairwise=True,
                exception=exc,
            )
        if ans.exception is None:
            self._dump_response(ans, self.output_columns, output_file)
        return ans

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

    def _prepare_documents(self, query: Query) -> str:
        documents = []
        for d in query.retrieved_docs:
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
                "did": d.did,
                "doc": d.text,
                "annotation": d.evaluation.answer if d.evaluation else None,
            }
            documents.append(self.config.document_template.format(**formatters))
        if len(documents) == 0:
            return "NO DOCUMENTS WERE RETRIEVED"
        return "\n".join(documents)

    def __prepare_queries(self, queries: list[Query]) -> list[Query]:
        queries = self._load_retrieved_documents(queries)
        queries = self._load_document_evaluations(queries, force=False)
        queries = self._load_agent_answers(queries)
        queries = self.__add_pairwise_games(queries)
        queries = self._load_answers_evaluations(queries, force=self.config.force)
        return queries

    def __add_pairwise_games(self, queries: list[Query]) -> list[Query]:
        if not self.config.pairwise:
            return queries
        for query in queries:
            query_agents = list({x.agent for x in query.answers})
            pairs = list(itertools.combinations(query_agents, 2))
            if not isinstance(self.config, PairwiseEvaluatorConfig):
                raise ValueError(
                    "You are trying to generate pairwise games for a pointwise evaluator"
                )
            if self.config.bidirectional:
                pairs += [(b, a) for a, b in pairs]
            random.shuffle(pairs)

            # Filter out games that already exist
            existing_games = {
                (a.agent_a_answer.agent, a.agent_b_answer.agent)
                for a in query.pairwise_games
            }
            answer_idx = {ans.agent: idx for idx, ans in enumerate(query.answers)}
            games = [g for g in pairs if g not in existing_games]

            games_to_add = self.config.n_games_per_query - len(existing_games)
            games = games[:games_to_add]
            for agent_a, agent_b in games:
                query.pairwise_games.append(
                    PairwiseGame(
                        agent_a_answer=query.answers[answer_idx[agent_a]],
                        agent_b_answer=query.answers[answer_idx[agent_b]],
                    )
                )
        return queries

    def __get_tuples_to_evaluate(
        self,
        queries: list[Query],
    ) -> list[tuple[Query, PairwiseGame | AgentAnswer]]:
        tuples_to_eval: list[tuple[Query, PairwiseGame | AgentAnswer]] = []
        all_tuples = 0
        for q in queries:
            if self.config.pairwise:
                for g in q.pairwise_games:
                    all_tuples += 1
                    if g.evaluation is None:
                        tuples_to_eval.append((q, g))
            else:
                for a in q.answers:
                    all_tuples += 1
                    if a.evaluation is None:
                        tuples_to_eval.append((q, a))
        if len(tuples_to_eval) == 0:
            logger.info("All answers have been evaluated")
            if self.config.verbose:
                print(
                    f"All {all_tuples} answers are already evaluated.\n"
                    "If you want to re-evaluate them, use the force flag"
                )

        return tuples_to_eval

    def batch_evaluate(self, queries: list[Query]) -> list[Query]:
        try:
            # Raises RuntimeError if there is no current event loop.
            asyncio.get_running_loop()
            # If there is a current event loop, we need to run the async code
            # in a separate loop, in a separate thread.
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    asyncio.run, self._async_batch_evaluate(queries)
                )
                result = future.result()
        except RuntimeError:
            result = asyncio.run(self._async_batch_evaluate(queries))

        return result


class AnswerEvaluatorFactory:
    registry: dict[AnswerEvaluatorTypes | str, Type[BaseAnswerEvaluator]] = {}

    @classmethod
    def register(cls, name: AnswerEvaluatorTypes) -> Callable:
        def inner_wrapper(wrapped_class: Type[BaseAnswerEvaluator]):
            if name in cls.registry:
                logger.warning(f"Overwriting {name} in registry")
            cls.registry[name.lower()] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        evaluator_name: str,
        llm_provider: BaseLLMProvider | str,
        config: Optional[BaseAnswerEvaluatorConfig] = None,
        **kwargs,
    ) -> BaseAnswerEvaluator:
        if evaluator_name.lower() not in cls.registry:
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
        return cls.registry[evaluator_name.lower()].from_config(
            config, llm_provider_instance
        )


def get_answer_evaluator(
    evaluator_name: Optional[AnswerEvaluatorTypes | str] = None,
    llm_provider: BaseLLMProvider | str = "openai",
    config: Optional[BaseAnswerEvaluatorConfig] = None,
    **kwargs,
) -> BaseAnswerEvaluator:
    if evaluator_name is None:
        # get the name from the config
        if config is None:
            raise ValueError(
                "Either the evaluator_name or a config object must be provided"
            )
        evaluator_name = config.evaluator_name
    return AnswerEvaluatorFactory.create(
        evaluator_name,
        llm_provider=llm_provider,
        config=config,
        **kwargs,
    )
