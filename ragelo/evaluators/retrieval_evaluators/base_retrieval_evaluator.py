"""A Retrieval Evaluator is a class that evaluates the results of a retrieval system.
It receives a set of queries used to retrieve a document and their respective retrieved documents,
and returns a score or a label for each document."""

import asyncio
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

from tqdm.auto import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types import (
    AnswerFormat,
    Document,
    Query,
    RetrievalEvaluatorResult,
    RetrievalEvaluatorTypes,
)
from ragelo.types.configurations import BaseRetrievalEvaluatorConfig


class BaseRetrievalEvaluator(BaseEvaluator):
    config: BaseRetrievalEvaluatorConfig
    output_columns = ["qid", "did", "raw_answer", "answer"]

    def __init__(
        self,
        config: BaseRetrievalEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.config = config
        self.llm_provider = llm_provider
        self.document_evaluations_path = config.document_evaluations_path
        if config.output_columns:
            self.output_columns = config.output_columns
        if config.answer_format == AnswerFormat.MULTI_FIELD_JSON:
            self.config.scoring_keys = config.scoring_keys
            self.output_columns = [
                "qid",
                "did",
                "raw_answer",
            ] + self.config.scoring_keys

        if config.scoring_key and config.scoring_key not in self.output_columns:
            print(f"Adding scoring key {config.scoring_key} to output columns")
            self.output_columns.append(self.config.scoring_key)
        if config.scoring_keys:
            missing_keys = [
                key for key in config.scoring_keys if key not in self.output_columns
            ]
            self.output_columns.extend(missing_keys)

    async def _async_batch_evaluate(self, queries: List[Query]) -> List[Query]:
        use_progress_bar = self.config.use_progress_bar
        queries = self.__prepare_queries(queries)
        tuples_to_eval = self.__get_tuples_to_evaluate(queries)
        if len(tuples_to_eval) == 0:
            return queries
        pbar = tqdm(
            total=len(tuples_to_eval),
            ncols=100,
            desc="Evaluating retrieved documents",
            disable=not use_progress_bar,
            leave=False,
            position=0,
        )
        awaitables_ended = False
        pending: Set[asyncio.Future] = set()
        aws = map(self._async_evaluate, tuples_to_eval)
        aws = iter(aws)
        evaluations = []
        failed = 0
        while pending or not awaitables_ended:
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
                    failed += 1
                    continue
                evaluations.append(evaluation)
        pbar.close()
        self._add_document_evaluations(queries, evaluations)
        if self.config.verbose:
            print("✅ Done!")
            print("Failed evaluations:", failed)
            print(f"Total evaluations: {len(evaluations)}")
        return queries

    async def _async_evaluate(
        self, eval_sample: Tuple[Query, Document]
    ) -> RetrievalEvaluatorResult:
        query, document = eval_sample
        exc = None
        prompt = self._build_message(query, document)
        try:
            raw_answer = await self.llm_provider.call_async(prompt)
        except Exception as e:
            logger.warning(f"Failed to FETCH answers for qid: {query.qid}")
            logger.warning(f"document id: {document.did}")
            exc = str(e)
            raw_answer = None
        try:
            answer = self._process_answer(raw_answer) if raw_answer else None
        except ValueError as e:
            logger.warning(
                f"Failed to PARSE answer for qid: {query.qid} "
                "document id: {document.did}\n"
                f"Raw answer: {raw_answer}"
            )
            exc = str(e)
            answer = None
        ans = RetrievalEvaluatorResult(
            qid=query.qid,
            did=document.did,
            raw_answer=raw_answer,
            answer=answer,
            exception=exc,
        )
        if ans.exception is None:
            self._dump_response(
                ans, self.output_columns, self.document_evaluations_path
            )
        return ans

    def __prepare_queries(self, queries: List[Query]) -> List[Query]:
        queries = self._load_retrieved_documents(queries)
        queries = self._load_document_evaluations(queries, force=self.config.force)
        return queries

    def __get_tuples_to_evaluate(
        self, queries: List[Query]
    ) -> List[Tuple[Query, Document]]:
        tuples_to_eval = []
        all_tuples = 0
        for q in queries:
            for d in q.retrieved_docs:
                if d.evaluation is None:
                    tuples_to_eval.append((q, d))
                all_tuples += 1
        if len(tuples_to_eval) == 0:
            logger.info("All documents have been evaluated")
            if self.config.verbose:
                print(
                    f"All {all_tuples} documents are already evaluated.\n"
                    "If you want to re-evaluate documents, use the --force flag."
                )
        return tuples_to_eval

    def evaluate(
        self,
        query: Union[Query, str],
        document: Union[Document, str],
        query_metadata: Optional[Dict[str, Any]] = None,
        doc_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any]:
        """Evaluates a single query-document pair. Returns the raw answer and the processed answer."""
        query = self._assemble_query(query, query_metadata)
        document = self._assemble_document(document, doc_metadata)

        def run(coroutine):
            return asyncio.run(coroutine)

        try:
            # Raises RuntimeError if there is no current event loop.
            asyncio.get_running_loop()
            # If there is a current event loop, we need to run the async code
            # in a separate loop, in a separate thread.
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run, self._async_evaluate((query, document)))
                result = future.result()
        except RuntimeError:
            result = asyncio.run(self._async_evaluate((query, document)))
        if result.exception or result.raw_answer is None or result.answer is None:
            raise ValueError(
                f"Failed to evaluate qid: {query.qid} did: {document.did}",
                f"Exception: {result.exception}",
            )
        return result.raw_answer, result.answer

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

    def _build_message(
        self,
        query: Query,
        document: Document,
    ) -> Union[str, List[Dict[str, str]]]:
        """Builds the prompt to send to the LLM."""
        raise NotImplementedError

    @classmethod
    def from_config(
        cls, config: BaseRetrievalEvaluatorConfig, llm_provider: BaseLLMProvider
    ):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> Type[BaseRetrievalEvaluatorConfig]:
        return get_type_hints(cls)["config"]

    @staticmethod
    def _construct_list_of_answers(
        answers: List[Dict[str, str]]
    ) -> List[RetrievalEvaluatorResult]:
        return [RetrievalEvaluatorResult(**x) for x in answers]


class RetrievalEvaluatorFactory:
    registry: Dict[RetrievalEvaluatorTypes, Type[BaseRetrievalEvaluator]] = {}

    @classmethod
    def register(cls, evaluator_name: RetrievalEvaluatorTypes) -> Callable:
        def inner_wrapper(
            wrapped_class: Type[BaseRetrievalEvaluator],
        ) -> Type[BaseRetrievalEvaluator]:
            if evaluator_name in cls.registry:
                logger.debug(f"Overwriting {evaluator_name} in registry")
            cls.registry[evaluator_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        evaluator_name: RetrievalEvaluatorTypes,
        llm_provider: Union[BaseLLMProvider, str],
        config: Optional[BaseRetrievalEvaluatorConfig] = None,
        **kwargs,
    ) -> BaseRetrievalEvaluator:
        if isinstance(llm_provider, str):
            llm_provider_instance = get_llm_provider(llm_provider, **kwargs)
        else:
            llm_provider_instance = llm_provider
        if evaluator_name not in cls.registry:
            raise ValueError(
                f"Unknown retrieval evaluator {evaluator_name}\n"
                f"Valid options are {list(cls.registry.keys())}"
            )
        if config is None:
            class_ = cls.registry[evaluator_name]
            type_config = class_.get_config_class()
            valid_keys = [field for field in type_config.get_model_fields()]
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            config = type_config(**valid_args)
        return cls.registry[evaluator_name].from_config(config, llm_provider_instance)


def get_retrieval_evaluator(
    evaluator_name: Optional[Union[RetrievalEvaluatorTypes, str]] = None,
    llm_provider: Union[BaseLLMProvider, str] = "openai",
    config: Optional[BaseRetrievalEvaluatorConfig] = None,
    **kwargs,
) -> BaseRetrievalEvaluator:
    if evaluator_name is None:
        # get the name from the config
        if config is None:
            raise ValueError(
                "Either the evaluator_name or a config object must be provided"
            )
        evaluator_name = config.evaluator_name
    if isinstance(evaluator_name, str):
        evaluator_name = RetrievalEvaluatorTypes(evaluator_name)
    return RetrievalEvaluatorFactory.create(
        evaluator_name,
        llm_provider=llm_provider,
        config=config,
        **kwargs,
    )
