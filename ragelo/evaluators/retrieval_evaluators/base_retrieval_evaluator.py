"""A Retrieval Evaluator is a class that evaluates the results of a retrieval system.
It receives a set of queries used to retrieve a document and their respective retrieved documents,
and returns a score or a label for each document."""

from __future__ import annotations

import logging
from typing import Any, Callable, get_type_hints

from tenacity import RetryError

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.types import LLMInputPrompt, Query, RetrievalEvaluatorResult
from ragelo.types.configurations import BaseRetrievalEvaluatorConfig
from ragelo.types.evaluables import Document, Evaluable
from ragelo.types.types import RetrievalEvaluatorTypes
from ragelo.utils import call_async_fn

logger = logging.getLogger(__name__)


class BaseRetrievalEvaluator(BaseEvaluator):
    """
    A base class for retrieval evaluators.
    """

    config: BaseRetrievalEvaluatorConfig
    evaluable_name: str = "Retrieved document"
    result_type: type[RetrievalEvaluatorResult] = RetrievalEvaluatorResult

    def __init__(self, config: BaseRetrievalEvaluatorConfig, llm_provider: BaseLLMProvider):
        super().__init__(config, llm_provider)

    def evaluate(
        self,
        query: Query | str,
        document: Document | str,
        query_metadata: dict[str, Any] | None = None,
        doc_metadata: dict[str, Any] | None = None,
    ) -> RetrievalEvaluatorResult:
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
        document = Document.assemble_document(document, query.qid, doc_metadata)
        result = call_async_fn(self.evaluate_async, (query, document))

        if result.exception or result.answer is None:
            raise ValueError(
                f"Failed to evaluate qid: {query.qid} did: {document.did}",
                f"Exception: {result.exception}",
            )
        return result

    async def evaluate_async(self, eval_sample: tuple[Query, Evaluable]) -> RetrievalEvaluatorResult:
        """
        Evaluates a single query-document pair asynchronously.
        Args:
            eval_sample (tuple[Query, Evaluable]): The query and document to evaluate.
        """
        query, document = eval_sample
        if not isinstance(document, Document):
            type_name = type(document).__name__
            raise ValueError(f"can't evaluate a {type_name} in a Retrieval Evaluator")

        exc = None
        evaluator_name = str(self.config.evaluator_name)
        if evaluator_name in document.evaluations and not self.config.force:
            cached_eval = document.evaluations[evaluator_name]
            if isinstance(cached_eval, RetrievalEvaluatorResult):
                return cached_eval

        # Get the answer schema type from the result_type's 'answer' field
        answer_field = self.result_type.model_fields.get("answer")
        if not answer_field or not answer_field.annotation:
            raise ValueError(f"Result type {self.result_type} does not have an 'answer' field with annotation")

        answer_type = answer_field.annotation
        # Handle Optional types (answer_type might be "SomeType | None")
        if hasattr(answer_type, "__args__"):
            # Get the first non-None type from the union
            answer_type = next((arg for arg in answer_type.__args__ if arg is not type(None)), answer_type)

        llm_input = self._build_message(query, document)
        parsed_answer = None
        raw_answer = ""
        try:
            llm_response = await self.llm_provider.call_async(
                input=llm_input,
                response_schema=answer_type,
            )
            llm_response = self._process_answer(llm_response)
            parsed_answer = llm_response.parsed_answer
            raw_answer = llm_response.raw_answer
        except Exception as e:
            if isinstance(e, RetryError):
                exc = str(e) + "\nLLM Error: \n" + str(e.last_attempt.exception())
            elif raw_answer:
                exc = str(e) + f"\nRaw answer: {raw_answer}"
            logger.warning(f"Failed to generate answer for qid: {query.qid} and document: {document.did}: {exc}")

        return self.result_type(
            qid=query.qid,
            did=document.did,
            evaluator_name=str(self.config.evaluator_name),
            answer=parsed_answer,
            exception=exc,
        )

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        context = {"query": query, "document": document}
        user_message = self.user_prompt.render(**context) if self.user_prompt else None
        system_prompt = self.system_prompt.render(**context) if self.system_prompt else None

        return LLMInputPrompt(
            system_prompt=system_prompt,
            user_message=user_message,
        )

    @classmethod
    def from_config(cls, config: BaseRetrievalEvaluatorConfig, llm_provider: BaseLLMProvider):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> type[BaseRetrievalEvaluatorConfig]:
        return get_type_hints(cls)["config"]

    def _get_all_evaluables(self, query: Query) -> list[Evaluable]:
        return list(query.retrieved_docs.values())


class RetrievalEvaluatorFactory:
    registry: dict[RetrievalEvaluatorTypes, type[BaseRetrievalEvaluator]] = {}

    @classmethod
    def register(cls, evaluator_name: RetrievalEvaluatorTypes) -> Callable:
        def inner_wrapper(
            wrapped_class: type[BaseRetrievalEvaluator],
        ) -> type[BaseRetrievalEvaluator]:
            if evaluator_name in cls.registry:
                logger.debug(f"Overwriting {evaluator_name} in registry")
            cls.registry[evaluator_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get_evaluator_result_type(cls, evaluator_name: RetrievalEvaluatorTypes) -> type[RetrievalEvaluatorResult]:
        """Gets the retrieval evaluator result type for a specific evaluator type.

        Args:
            evaluator_name (RetrievalEvaluatorTypes): The name of the evaluator.

        Returns:
            type: The retrieval evaluator result type for the evaluator.
        """
        if evaluator_name not in cls.registry:
            raise ValueError(
                f"Unknown retrieval evaluator {evaluator_name}\nValid options are {list(cls.registry.keys())}"
            )
        evaluator_class = cls.registry[evaluator_name]
        return evaluator_class.result_type

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
                f"Unknown retrieval evaluator {evaluator_name}\nValid options are {list(cls.registry.keys())}"
            )
        if isinstance(llm_provider, str):
            llm_provider_instance = get_llm_provider(llm_provider, **kwargs)
        else:
            llm_provider_instance = llm_provider
        if config is None:
            class_ = cls.registry[evaluator_name]
            type_config = class_.get_config_class()
            valid_keys = [field for field in type_config.model_fields]
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            required_fields = [arg for arg, info in type_config.model_fields.items() if info.is_required()]
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
        try:
            evaluator_name = RetrievalEvaluatorTypes(evaluator_name)
        except ValueError:
            raise ValueError(f"Unknown retrieval evaluator {evaluator_name}")
    if evaluator_name is None:
        raise ValueError("The evaluator_name must be provided")
    return RetrievalEvaluatorFactory.create(
        evaluator_name,
        llm_provider=llm_provider,
        config=config,
        **kwargs,
    )


def get_retrieval_evaluator_result_type(
    evaluator_name: RetrievalEvaluatorTypes | str,
) -> type[RetrievalEvaluatorResult]:
    """Gets the retrieval evaluator result type for a specific evaluator type.

    Args:
        evaluator_name (RetrievalEvaluatorTypes | str): The name of the retrieval evaluator.

    Returns:
        type: The retrieval evaluator result type for the evaluator.
    """
    if isinstance(evaluator_name, str):
        try:
            evaluator_name = RetrievalEvaluatorTypes(evaluator_name)
        except ValueError:
            raise ValueError(f"Unknown retrieval evaluator {evaluator_name}")
    return RetrievalEvaluatorFactory.get_evaluator_result_type(evaluator_name)
