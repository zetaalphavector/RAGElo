"""A Retrieval Evaluator is a class that evaluates the results of a retrieval system.
It receives a set of queries used to retrieve a document and their respective retrieved documents,
and returns a score or a label for each document."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, get_type_hints

from pydantic import BaseModel, Field, create_model

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider, split_llm_provider_kwargs
from ragelo.types import LLMInputPrompt, Query, RetrievalEvaluatorResult
from ragelo.types.answer_formats import EvaluationAnswer, RetrievalEvaluationAnswer
from ragelo.types.configurations import BaseRetrievalEvaluatorConfig
from ragelo.types.evaluables import Document, Evaluable
from ragelo.types.evaluator_utils import answer_format_for
from ragelo.types.types import RetrievalEvaluatorTypes, _result_type_registry
from ragelo.utils import call_async_fn, describe_exception, warn_ignored_arguments

logger = logging.getLogger(__name__)

T_Config = TypeVar("T_Config", bound=BaseRetrievalEvaluatorConfig)

if TYPE_CHECKING:
    from ragelo.types.experiment import Experiment


class BaseRetrievalEvaluator(BaseEvaluator[T_Config, RetrievalEvaluatorResult]):
    """
    A base class for retrieval evaluators.
    """

    config: T_Config
    evaluable_name: str = "Retrieved document"
    result_type: type[RetrievalEvaluatorResult] = RetrievalEvaluatorResult
    answer_format: type[EvaluationAnswer] = RetrievalEvaluationAnswer
    relevance_grades: Sequence[str] = ("non-relevant", "somewhat relevant", "highly relevant")

    def __init__(self, config: T_Config, llm_provider: BaseLLMProvider):
        super().__init__(config, llm_provider)
        states_its_grades = type(self).relevance_grades is not BaseRetrievalEvaluator.relevance_grades
        if config.relevance_grades:
            self.relevance_grades = config.relevance_grades
        # A custom prompt brings its own scale, so its score is only bounded once the grades are given.
        if self.answer_format is RetrievalEvaluationAnswer and (states_its_grades or config.relevance_grades):
            self.answer_format = create_model(
                "RetrievalEvaluationAnswer", __base__=RetrievalEvaluationAnswer, score=(int, self._score_field())
            )

    @property
    def max_score(self) -> int:
        return len(self.relevance_grades) - 1

    def _score_field(self) -> Any:
        grades = "\n".join(f"{score}: {grade}" for score, grade in enumerate(self.relevance_grades))
        return Field(
            description=f"Your relevance score for the document, an integer from 0 to {self.max_score}.\n{grades}",
            ge=0,
            le=self.max_score,
        )

    def _prompt_context(self, query: Query, document: Document) -> dict[str, Any]:
        """`custom_grades` lets a prompt drop text that only explains the evaluator's own grades."""
        return {
            "query": query,
            "document": document,
            "relevance_grades": self.relevance_grades,
            "max_score": self.max_score,
            "custom_grades": self.config.relevance_grades is not None,
        }

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
        query = Query.build(query, query_metadata)
        document = Document.build(document, query.qid, metadata=doc_metadata)
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
            raise TypeError(f"can't evaluate a {type_name} in a Retrieval Evaluator")

        exc = None
        evaluator_name = str(self.config.evaluator_name)
        if evaluator_name in document.evaluations and not self.config.force:
            cached_eval = document.evaluations[evaluator_name]
            if isinstance(cached_eval, RetrievalEvaluatorResult) and self._is_cached_result_valid(query, cached_eval):
                return cached_eval

        llm_input = self._with_guidelines(self._build_message(query, document))
        answer_type = self._resolve_response_schema(llm_input)
        parsed_answer = None
        usage = None
        try:
            llm_response = await self.llm_provider.call_async(
                input=llm_input,
                response_schema=answer_type,
            )
            usage = llm_response.usage
            llm_response = self._process_answer(llm_response, query)
            parsed_answer = llm_response.parsed_answer
        except Exception as e:  # noqa: BLE001
            exc = describe_exception(e)
            logger.warning(f"Failed to generate answer for qid: {query.qid} and document: {document.did}: {exc}")

        return self.result_type(
            qid=query.qid,
            did=document.did,
            evaluator_name=str(self.config.evaluator_name),
            answer=parsed_answer,  # type: ignore[arg-type]
            exception=exc,
            usage=usage,
        )

    def _get_all_evaluables(self, query: Query) -> list[Evaluable]:
        return list(query.retrieved_docs.values())

    def _get_tuples_to_evaluate(self, experiment: Experiment) -> Sequence[tuple[Query, Evaluable]]:
        """
        Creates the list of pairs (query, evaluable) to evaluate
        """
        tuples_to_eval = []
        all_tuples = 0
        missing_evaluations = 0
        evaluator_name = str(self.config.evaluator_name)
        for q in experiment:
            for d in q.retrieved_docs.values():
                if evaluator_name not in d.evaluations:
                    missing_evaluations += 1
                tuples_to_eval.append((q, d))
                all_tuples += 1
        if missing_evaluations == 0 and not self.config.force:
            logger.info(
                f"All {all_tuples} documents are already evaluated.\n"
                "If you want to re-evaluate them, use the --force flag"
            )
        return tuples_to_eval

    def _resolve_response_schema(self, prompt: LLMInputPrompt) -> type[BaseModel]:
        """The schema the LLM is asked to produce, most specific source first.

        Evaluators whose response shape depends on the query (a rubric with one field per criterion)
        set `llm_response_schema` on the prompt they build; the rest declare `answer_format`.
        """
        schema = prompt.llm_response_schema or self.config.llm_response_schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema
        if self.config.result_type:
            return self.config.result_type
        return answer_format_for(self, BaseRetrievalEvaluator)

    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        context = self._prompt_context(query, document)
        user_message = self.user_prompt.render(**context) if self.user_prompt else None
        system_prompt = self.system_prompt.render(**context) if self.system_prompt else None

        return LLMInputPrompt(
            system_prompt=system_prompt,
            user_message=user_message,
        )

    @classmethod
    def from_config(cls, config: T_Config, llm_provider: BaseLLMProvider):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> type[BaseRetrievalEvaluatorConfig]:
        return get_type_hints(cls)["config"]


class RetrievalEvaluatorFactory:
    registry: ClassVar[dict[RetrievalEvaluatorTypes, type[BaseRetrievalEvaluator]]] = {}

    @classmethod
    def register(cls, evaluator_name: RetrievalEvaluatorTypes) -> Callable:
        def inner_wrapper(
            wrapped_class: type[BaseRetrievalEvaluator],
        ) -> type[BaseRetrievalEvaluator]:
            if evaluator_name in cls.registry:
                logger.debug(f"Overwriting {evaluator_name} in registry")
            cls.registry[evaluator_name] = wrapped_class
            _result_type_registry[f"retrieval:{evaluator_name}"] = wrapped_class.result_type
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
            provider_kwargs, kwargs = split_llm_provider_kwargs(llm_provider, kwargs)
            llm_provider_instance = get_llm_provider(llm_provider, **provider_kwargs)
        else:
            llm_provider_instance = llm_provider
        if config is None:
            config_class = cls.registry[evaluator_name].get_config_class()
            warn_ignored_arguments(f"The {evaluator_name} retrieval evaluator", config_class, kwargs)
            config = config_class(**kwargs)
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
