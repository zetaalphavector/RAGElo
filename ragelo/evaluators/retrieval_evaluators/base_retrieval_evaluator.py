"""A Retrieval Evaluator is a class that evaluates the results of a retrieval system.
It receives a set of queries used to retrieve a document and their respective retrieved documents,
and returns a score or a label for each document."""

from __future__ import annotations

from typing import Any, Callable, Type, get_type_hints

from tenacity import RetryError

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types.configurations import BaseRetrievalEvaluatorConfig
from ragelo.types.evaluables import Document, Evaluable
from ragelo.types.experiment import Experiment
from ragelo.types.formats import LLMResponseType
from ragelo.types.query import Query
from ragelo.types.results import RetrievalEvaluatorResult
from ragelo.types.types import RetrievalEvaluatorTypes
from ragelo.utils import call_async_fn


class BaseRetrievalEvaluator(BaseEvaluator):
    config: BaseRetrievalEvaluatorConfig
    evaluable_name: str = "Retrieved document"

    def __init__(
        self,
        config: BaseRetrievalEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.config = config
        self.llm_provider = llm_provider

    def evaluate(
        self,
        query: Query | str,
        document: Document | str,
        query_metadata: dict[str, Any] | None = None,
        doc_metadata: dict[str, Any] | None = None,
    ) -> RetrievalEvaluatorResult:
        """Evaluates a single query-document pair. Returns the raw answer and the processed answer."""
        query = Query.assemble_query(query, query_metadata)
        document = Document.assemble_document(document, query.qid, doc_metadata)
        result = call_async_fn(self.evaluate_async, (query, document))

        if result.exception or result.raw_answer is None or result.answer is None:
            raise ValueError(
                f"Failed to evaluate qid: {query.qid} did: {document.did}",
                f"Exception: {result.exception}",
            )
        return result

    async def evaluate_async(
        self,
        eval_sample: tuple[Query, Evaluable],
    ) -> RetrievalEvaluatorResult:
        query, document = eval_sample
        if not isinstance(document, Document):
            type_name = type(document).__name__
            raise ValueError(f"can't evaluate a {type_name} in a Retrieval Evaluator")
        exc = None
        if document.evaluation is not None and not self.config.force:
            return RetrievalEvaluatorResult(
                did=document.did,
                qid=query.qid,
                raw_answer=document.evaluation.raw_answer,
                answer=document.evaluation.answer,
                exception=document.evaluation.exception,
            )
        prompt = self._build_message(query, document)
        try:
            llm_response = await self.llm_provider.call_async(
                prompt,
                answer_format=self.config.llm_answer_format,
                response_schema=self.config.llm_response_schema,
            )
            llm_response = self._process_answer(llm_response)
        except ValueError as e:
            logger.warning(f"Failed to PARSE answer for qid: {query.qid} document id: {document.did}")
            try:
                llm_response = LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=None)
            except Exception:
                llm_response = LLMResponseType(raw_answer="", parsed_answer=None)
            exc = str(e)
        except Exception as e:
            logger.warning(f"Failed to FETCH answers for qid: {query.qid}")
            logger.warning(f"document id: {document.did}")
            if isinstance(e, RetryError):
                exc = str(e.last_attempt.exception())
            else:
                exc = str(e)
            try:
                llm_response = LLMResponseType(raw_answer=llm_response.raw_answer, parsed_answer=None)
            except Exception:
                llm_response = LLMResponseType(raw_answer="", parsed_answer=None)
        return RetrievalEvaluatorResult(
            qid=query.qid,
            did=document.did,
            raw_answer=llm_response.raw_answer,
            answer=llm_response.parsed_answer,
            exception=exc,
        )

    def _get_tuples_to_evaluate(self, experiment: Experiment) -> list[tuple[Query, Evaluable]]:
        """
        Creates the list of pairs (query, evaluable) to evaluate
        """
        tuples_to_eval = []
        all_tuples = 0
        missing_evaluations = 0
        for q in experiment:
            for d in q.retrieved_docs.values():
                if d.evaluation is None:
                    missing_evaluations += 1
                tuples_to_eval.append((q, d))
                all_tuples += 1
        if missing_evaluations == 0 and not self.config.force:
            logger.info(
                f"All {all_tuples} documents are already evaluated.\n"
                "If you want to re-evaluate them, use the --force flag"
            )
        return tuples_to_eval

    def _build_message(
        self,
        query: Query,
        document: Document,
    ) -> str | list[dict[str, str]]:
        """Builds the prompt to send to the LLM."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: BaseRetrievalEvaluatorConfig, llm_provider: BaseLLMProvider):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> Type[BaseRetrievalEvaluatorConfig]:
        return get_type_hints(cls)["config"]


class RetrievalEvaluatorFactory:
    registry: dict[RetrievalEvaluatorTypes, Type[BaseRetrievalEvaluator]] = {}

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
        llm_provider: BaseLLMProvider | str,
        config: BaseRetrievalEvaluatorConfig | None = None,
        **kwargs,
    ) -> BaseRetrievalEvaluator:
        if evaluator_name not in cls.registry:
            raise ValueError(
                f"Unknown retrieval evaluator {evaluator_name}\n" f"Valid options are {list(cls.registry.keys())}"
            )
        if isinstance(llm_provider, str):
            llm_provider_instance = get_llm_provider(llm_provider, **kwargs)
        else:
            llm_provider_instance = llm_provider
        if config is None:
            class_ = cls.registry[evaluator_name]
            type_config = class_.get_config_class()
            valid_keys = [field for field in type_config.get_model_fields()]
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            required_fields = [arg for arg, info in type_config.get_model_fields().items() if info.is_required()]
            for field in required_fields:
                if field not in valid_args:
                    raise ValueError(f"Required argument {field} for evaluator {evaluator_name} not provided")
            config = type_config(**valid_args)
        return cls.registry[evaluator_name].from_config(config, llm_provider_instance)


def get_retrieval_evaluator(
    evaluator_name: RetrievalEvaluatorTypes | str | None = None,
    llm_provider: BaseLLMProvider | str = "openai",
    config: BaseRetrievalEvaluatorConfig | None = None,
    **kwargs,
) -> BaseRetrievalEvaluator:
    if evaluator_name is None:
        # get the name from the config
        if config is None:
            raise ValueError("Either the evaluator_name or a config object must be provided")
        evaluator_name = config.evaluator_name
    if isinstance(evaluator_name, str):
        evaluator_name = RetrievalEvaluatorTypes(evaluator_name)
    return RetrievalEvaluatorFactory.create(
        evaluator_name,
        llm_provider=llm_provider,
        config=config,
        **kwargs,
    )
