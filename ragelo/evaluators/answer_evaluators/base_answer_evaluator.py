"""Base model for dealing with answer evaluators"""

import asyncio
import itertools
import random
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    get_type_hints,
)

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
        self.output_columns = (
            config.output_columns_answer_evaluator
            if not self.config.pairwise
            else config.output_columns_pairwise_evaluator
        )
        self.scoring_keys = config.scoring_keys_answer_evaluator
        self.scoring_key = config.scoring_key_answer_evaluator
        if isinstance(self.config.answer_format_answer_evaluator, str):
            self.answer_format = AnswerFormat(
                self.config.answer_format_answer_evaluator
            )
        else:
            self.answer_format = self.config.answer_format_answer_evaluator

        if self.answer_format == AnswerFormat.MULTI_FIELD_JSON:
            missing_keys = [
                key for key in self.scoring_keys if key not in self.output_columns
            ]
            self.output_columns.extend(missing_keys)
        else:
            if self.scoring_key not in self.output_columns:
                logger.info(f"Adding scoring key {self.scoring_key} to output columns")
                self.output_columns.append(self.scoring_key)

        self.pairwise = self.config.pairwise

    async def _async_batch_evaluate(self, queries: List[Query]) -> List[Query]:
        use_progress_bar = self.config.use_progress_bar
        failed_queries = 0
        queries = self.__prepare_queries(queries)
        tuples_to_eval = self.__get_tuples_to_evaluate(queries)

        if len(tuples_to_eval) == 0:
            return queries
        if self.config.rich_print:
            import warnings

            from tqdm import TqdmExperimentalWarning

            warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
            from tqdm.rich import tqdm as rich_tqdm

            pbar_fn = rich_tqdm  # type: ignore
        else:
            pbar_fn = tqdm  # type: ignore
        pbar = pbar_fn(
            total=len(tuples_to_eval),
            ncols=100,
            desc="Evaluating answers",
            disable=not use_progress_bar,
            leave=False,
            position=0,
        )

        awaitables_ended = False
        pending: Set[asyncio.Future] = set()
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
            self._print_failed_evaluations(len(evaluations), failed_queries)
        return queries

    def evaluate(
        self,
        query: Union[Query, str],
        answer: Optional[Union[AgentAnswer, str]] = None,
        answer_a: Optional[Union[AgentAnswer, str]] = None,
        answer_b: Optional[Union[AgentAnswer, str]] = None,
        retrieved_documents: Optional[Union[List[str], List[Document]]] = None,
        document_metadata: Optional[List[Dict[str, Any]]] = None,
        query_metadata: Optional[Dict[str, Any]] = None,
        answer_metadata=None,
        answer_a_metadata: Optional[Dict[str, Any]] = None,
        answer_b_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        query = self._assemble_query(query, query_metadata)
        if isinstance(retrieved_documents, str):
            retrieved_documents = [retrieved_documents]
        if retrieved_documents:
            retrieved_and_assembled_docs = self._assemble_documents(
                retrieved_documents, document_metadata
            )
            query.retrieved_docs = retrieved_and_assembled_docs
        agent: Union[str, Tuple[str, str]]
        if self.pairwise:
            if not answer_a or not answer_b:
                raise ValueError("Pairwise evaluations require two answers")
            answer_a = self._assemble_answer(answer_a, answer_a_metadata)
            answer_b = self._assemble_answer(answer_b, answer_b_metadata)
            agent = (answer_a.agent, answer_b.agent)
            game = PairwiseGame(
                agent_a_answer=answer_a,
                agent_b_answer=answer_b,
            )

            def run(coroutine):
                return asyncio.run(coroutine)

            try:
                asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run, self._async_evaluate((query, game)))
                    result = future.result()
            except RuntimeError:
                result = asyncio.run(self._async_evaluate((query, game)))
        else:
            if not answer:
                raise ValueError("Pointwise evaluations require an answer")
            answer = self._assemble_answer(answer, answer_metadata)
            agent = answer.agent

            def run(coroutine):
                return asyncio.run(coroutine)

            try:
                asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run, self._async_evaluate((query, answer)))
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
        self, eval_sample: Tuple[Query, Union[PairwiseGame, AgentAnswer]]
    ) -> AnswerEvaluatorResult:
        query, evaluable = eval_sample
        agent: Union[str, Tuple[str, str]]
        if evaluable.evaluation is not None and not self.config.force:
            return evaluable.evaluation  # type: ignore
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
            output_file = self.config.answers_evaluations_file
            ans = AnswerEvaluatorResult(
                qid=query.qid,
                agent=evaluable.agent,
                raw_answer=raw_answer,
                answer=answer,
                pairwise=False,
                exception=exc,
            )
        else:
            output_file = self.config.games_evaluations_file
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
    ) -> Union[str, List[Dict[str, str]]]:
        """Builds the message to send to the LLM evaluator"""
        raise NotImplementedError

    def _build_message_pairwise(
        self, query: Query, game: PairwiseGame
    ) -> Union[str, List[Dict[str, str]]]:
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

    def __prepare_queries(self, queries: List[Query]) -> List[Query]:
        queries = self._load_retrieved_documents(queries)
        queries = self._load_document_evaluations(queries, force=False)
        queries = self._load_agent_answers(queries)
        queries = self.__add_pairwise_games(queries)
        queries = self._load_answers_evaluations(queries, force=self.config.force)
        return queries

    def __add_pairwise_games(self, queries: List[Query]) -> List[Query]:
        if not self.pairwise:
            return queries
        for query in queries:
            query_agents = list(query.answers.keys())
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
        return queries

    def __get_tuples_to_evaluate(
        self, queries: List[Query]
    ) -> List[Tuple[Query, Union[PairwiseGame, AgentAnswer]]]:
        tuples_to_eval: List[Tuple[Query, Union[PairwiseGame, AgentAnswer]]] = []
        all_tuples = 0
        missing_evaluations = 0
        for q in queries:
            if self.pairwise:
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

        if missing_evaluations == all_tuples:
            logger.info("All answers have been evaluated")
            if self.config.verbose and not self.config.force:
                print(
                    f"All {all_tuples} answers are already evaluated.\n"
                    "If you want to re-evaluate them, use the force flag"
                )

        return tuples_to_eval

    def batch_evaluate(self, queries: List[Query]) -> List[Query]:
        def run(coroutine):
            return asyncio.run(coroutine)

        try:
            # Raises RuntimeError if there is no current event loop.
            asyncio.get_running_loop()
            # If there is a current event loop, we need to run the async code
            # in a separate loop, in a separate thread.
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run, self._async_batch_evaluate(queries))
                result = future.result()
        except RuntimeError:
            result = asyncio.run(self._async_batch_evaluate(queries))

        return result


class AnswerEvaluatorFactory:
    registry: Dict[AnswerEvaluatorTypes, Type[BaseAnswerEvaluator]] = {}

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
        llm_provider: Union[BaseLLMProvider, str],
        config: Optional[BaseAnswerEvaluatorConfig] = None,
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
    evaluator_name: Optional[Union[AnswerEvaluatorTypes, str]] = None,
    llm_provider: Union[BaseLLMProvider, str] = "openai",
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
    if isinstance(evaluator_name, str):
        evaluator_name = AnswerEvaluatorTypes(evaluator_name)
    return AnswerEvaluatorFactory.create(
        evaluator_name,
        llm_provider=llm_provider,
        config=config,
        **kwargs,
    )
