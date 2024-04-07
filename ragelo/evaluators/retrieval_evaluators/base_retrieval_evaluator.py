"""A Retrieval Evaluator is a class that evaluates the results of a retrieval system.
It receives a set of queries used to retrieve a document and their respective retrieved documents,
and returns a score or a label for each document."""

import csv
import dataclasses
import os
from abc import abstractmethod
from typing import Any, Callable, Optional, Type, get_type_hints

from tenacity import RetryError
from tqdm.auto import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types import (
    Document,
    Query,
    RetrievalEvaluatorResult,
    RetrievalEvaluatorTypes,
)
from ragelo.types.configurations import BaseEvaluatorConfig


class BaseRetrievalEvaluator(BaseEvaluator):
    config: BaseEvaluatorConfig
    output_columns: list[str] = ["qid", "did", "raw_answer", "answer"]
    scoring_key: str = "answer"
    output_file: str = "retrieval_evaluations.csv"
    tuple_columns: list[str] = ["qid", "did"]

    def __init__(
        self,
        config: BaseEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.config = config
        self.llm_provider = llm_provider
        if config.output_file is not None:
            self.output_file = config.output_file

    def batch_evaluate(self, queries: list[Query]) -> list[RetrievalEvaluatorResult]:
        """Evaluate all the documents for a list of queries"""
        use_progress_bar = self.config.verbose
        skip_docs = self._get_skip_tuples(
            self.output_file, self.tuple_columns, self.config.force
        )
        answers: list[RetrievalEvaluatorResult] = []
        failed_evaluations = 0
        for query in tqdm(
            queries,
            desc="Evaluating retrieved documents",
            disable=not use_progress_bar,
            ncols=100,
            leave=False,
            position=0,
        ):
            for document in tqdm(
                query.retrieved_docs,
                desc=query.qid,
                disable=not use_progress_bar,
                ncols=100,
                leave=False,
                position=1,
            ):
                qid = query.qid
                did = document.did
                if (qid, did) in skip_docs:
                    logger.debug(f"Skipping {qid} {did}")
                    continue
                try:
                    raw_answer, answer = self.evaluate(query, document)
                except (RetryError, ValueError):
                    failed_evaluations += 1
                    continue

                answers.append(
                    RetrievalEvaluatorResult(
                        qid=qid,
                        did=did,
                        raw_answer=raw_answer,
                        answer=answer,
                    )
                )
            self._dump_response(answers[-1], self.output_columns, self.output_file)

        if self.config.verbose:
            print("✅ Done!")
            print(f"Unparsed answers: {failed_evaluations}")
            print(f"Total evaluations: {len(answers)}")
        return answers

    def evaluate(
        self,
        query: Query | str,
        document: Document | str,
        query_metadata: Optional[dict[str, Any]] = None,
        doc_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Any]:
        """Evaluates a single query-document pair. Returns the raw answer and the processed answer."""
        if isinstance(query, str):
            query = Query(qid="<no_qid>", query=query)
        if isinstance(document, str):
            document = Document(did="<no_did>", text=document)
        if query_metadata:
            if query.metadata is not None:
                logger.warning(
                    f"Query metadata for query id {query.qid} is being overwritten!\n"
                    f"Old metadata: {query.metadata}\n"
                    f"New metadata: {query_metadata}\n"
                )
            query.metadata = query_metadata
        if doc_metadata:
            if document.metadata is not None:
                logger.warning(
                    f"Document metadata for document id {document.did} is being overwritten!\n"
                    f"Old metadata: {document.metadata}\n"
                    f"New metadata: {doc_metadata}\n"
                )
            document.metadata = doc_metadata

        message = self._build_message(query, document)
        try:
            raw_answer = self.llm_provider(message)
        except RetryError as e:
            logger.warning(
                f"Failed to FETCH answers for qid: {query.qid} did: {document.did}"
            )
            raise e
        try:
            answer = self._process_answer(raw_answer)
        except ValueError as e:
            logger.warning(
                f"Failed to PARSE answer for qid: {query.qid} did: {document.did}"
            )
            raise e
        return raw_answer, answer

    @abstractmethod
    def _build_message(
        self, query: Query, document: Document
    ) -> str | list[dict[str, str]]:
        """Builds the prompt to send to the LLM."""
        raise NotImplementedError

    @abstractmethod
    def _process_answer(self, answer: str) -> Any:
        """Processes the LLM evaluator output into some serializable format"""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: BaseEvaluatorConfig, llm_provider: BaseLLMProvider):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> Type[BaseEvaluatorConfig]:
        return get_type_hints(cls)["config"]


class RetrievalEvaluatorFactory:
    registry: dict[RetrievalEvaluatorTypes | str, Type[BaseRetrievalEvaluator]] = {}

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
        evaluator_name: RetrievalEvaluatorTypes | str,
        llm_provider: BaseLLMProvider | str,
        config: Optional[BaseEvaluatorConfig] = None,
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
            valid_keys = [field.name for field in dataclasses.fields(type_config)]
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            config = type_config(**valid_args)
        return cls.registry[evaluator_name].from_config(config, llm_provider_instance)


def get_retrieval_evaluator(
    evaluator_name: RetrievalEvaluatorTypes | str,
    llm_provider: BaseLLMProvider | str,
    config: Optional[BaseEvaluatorConfig] = None,
    **kwargs,
) -> BaseRetrievalEvaluator:
    return RetrievalEvaluatorFactory.create(
        evaluator_name,
        llm_provider=llm_provider,
        config=config,
        **kwargs,
    )
