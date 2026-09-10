from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Generic, TypeVar

from jinja2 import Template
from pydantic import BaseModel, create_model

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.presenters import render_failed_evaluations
from ragelo.types.answer_formats import RubricJudgment
from ragelo.types.configurations import BaseEvaluatorConfig
from ragelo.types.evaluables import Evaluable
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.query import Query
from ragelo.types.results import EvaluatorResult
from ragelo.utils import call_async_fn, get_pbar, string_to_template, with_guidelines

if TYPE_CHECKING:
    from ragelo.types.experiment import Experiment

T_Config = TypeVar("T_Config", bound=BaseEvaluatorConfig)
T_Result = TypeVar("T_Result", bound=EvaluatorResult)


class BaseEvaluator(ABC, Generic[T_Config, T_Result]):
    """
    An abstract class for all evaluators. An evaluator is responsible for evaluating a query and an evaluable
    """

    config: T_Config
    system_prompt: Template | None = None
    user_prompt: Template
    evaluable_name: str = "Evaluable"
    result_type: type[T_Result]

    def __init__(self, config: T_Config, llm_provider: BaseLLMProvider):
        self.config = config
        if config.result_type:
            self.result_type = create_model(
                config.result_type.__name__, answer=(config.result_type, ...), __base__=self.result_type
            )
        elif not hasattr(self, "result_type"):
            raise ValueError(f"Result format not set for evaluator {self.config.evaluator_name}")
        self.llm_provider = llm_provider
        if isinstance(config.system_prompt, Template):
            self.system_prompt = config.system_prompt
        elif isinstance(config.system_prompt, str):
            self.system_prompt = string_to_template(config.system_prompt)
        if isinstance(config.user_prompt, Template):
            self.user_prompt = config.user_prompt
        if isinstance(config.user_prompt, str):
            self.user_prompt = string_to_template(config.user_prompt)

    def _with_guidelines(self, prompt: LLMInputPrompt) -> LLMInputPrompt:
        if not self.config.guidelines:
            return prompt
        return prompt.model_copy(
            update={"system_prompt": with_guidelines(prompt.system_prompt, self.config.guidelines)}
        )

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
        self.prepare_experiment(experiment)
        call_async_fn(self._evaluate_experiment_async, experiment, n_threads)

    def evaluate_all_evaluables(self, query: Query, n_threads: int | None = None):
        """Evaluate all evaluables for a single query, useful for incremental workflows.
        Args:
            query: The query whose evaluables should be evaluated.
            n_threads: Maximum concurrent LLM calls. Defaults to ``config.n_processes``.
        """
        n_threads = n_threads or self.config.n_processes
        self.prepare_query(query)
        call_async_fn(self._evaluate_all_evaluables_async, query, n_threads)

    async def _evaluate_all_evaluables_async(self, query: Query, n_threads: int):
        tuples_to_eval = [(query, e) for e in self._get_all_evaluables(query)]
        pbar = get_pbar(
            len(tuples_to_eval),
            self.config.rich_print,
            desc=f"Evaluating {self.evaluable_name}s for query {query.qid}",
        )
        failed = 0
        evaluations = 0
        async for eval_tuple, evaluation in self._run_evaluations(tuples_to_eval, n_threads):
            evaluations += 1
            pbar.update()
            if evaluation.exception:
                failed += 1
                continue
            query.add_evaluation(eval_tuple[1], evaluation, exist_ok=True)
        pbar.close()
        if self.config.show_results:
            render_failed_evaluations(evaluations, failed, self.config.rich_print)

    def prepare_experiment(self, experiment: Experiment) -> None:
        """Produce the artifacts this evaluator grades against, before any judging starts."""
        for query in experiment:
            self.prepare_query(query)

    def prepare_query(self, query: Query) -> None:
        """Produce a single query's artifacts. Evaluators that grade against data they can generate
        override this; the rest need nothing.
        """
        return

    @abstractmethod
    def _get_all_evaluables(self, query: Query) -> list[Evaluable]:
        """Returns all evaluables for a given query"""
        raise NotImplementedError

    @abstractmethod
    async def evaluate_async(self, eval_sample: tuple[Query, Evaluable]) -> EvaluatorResult:
        """Evaluate a single query and evaluable asynchronously."""
        raise NotImplementedError

    def run_evaluations(
        self, tuples_to_eval: Sequence[tuple[Query, Evaluable]], n_threads: int | None = None
    ) -> list[tuple[tuple[Query, Evaluable], EvaluatorResult]]:
        n_threads = n_threads or self.config.n_processes
        return call_async_fn(self._collect_evaluations, tuples_to_eval, n_threads)

    async def _collect_evaluations(
        self, tuples_to_eval: Sequence[tuple[Query, Evaluable]], n_threads: int
    ) -> list[tuple[tuple[Query, Evaluable], EvaluatorResult]]:
        return [pair async for pair in self._run_evaluations(tuples_to_eval, n_threads)]

    async def _run_evaluations(
        self, tuples_to_eval: Sequence[tuple[Query, Evaluable]], n_threads: int
    ) -> AsyncIterator[tuple[tuple[Query, Evaluable], EvaluatorResult]]:
        pending: set[asyncio.Future] = set()
        tuples_iter = iter(tuples_to_eval)
        future_to_tuple: dict[asyncio.Future, tuple[Query, Evaluable]] = {}
        awaitables_ended = False
        while pending or not awaitables_ended:
            while len(pending) < n_threads and not awaitables_ended:
                try:
                    eval_tuple = next(tuples_iter)
                except StopIteration:
                    awaitables_ended = True
                else:
                    future = asyncio.ensure_future(self.evaluate_async(eval_tuple))
                    pending.add(future)
                    future_to_tuple[future] = eval_tuple
            if not pending:
                break
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            while done:
                finished = done.pop()
                evaluation = await finished
                eval_tuple = future_to_tuple.pop(finished)
                yield eval_tuple, evaluation

    async def _evaluate_experiment_async(self, experiment: Experiment, n_threads: int = 1):
        tuples_to_eval = self._get_tuples_to_evaluate(experiment)
        if len(tuples_to_eval) == 0:
            return
        pbar = get_pbar(
            len(tuples_to_eval),
            self.config.rich_print,
            desc=f"Evaluating {self.evaluable_name}s",
            disable=not getattr(self.config, "use_progress_bar", True),
        )
        failed = 0
        evaluations = 0
        async for eval_tuple, evaluation in self._run_evaluations(tuples_to_eval, n_threads):
            evaluations += 1
            pbar.update()
            pbar.refresh()
            if evaluation.exception:
                failed += 1
                continue
            experiment.add_evaluation(
                eval_tuple,
                evaluation,
                exist_ok=True,
                force=self.config.force,
                should_print=self.config.show_results,
            )
        pbar.close()
        if self.config.show_results:
            render_failed_evaluations(evaluations, failed, self.config.rich_print)

    @abstractmethod
    def _get_tuples_to_evaluate(self, experiment: Experiment) -> Sequence[tuple[Query, Evaluable]]:
        raise NotImplementedError

    def _is_cached_result_valid(self, query: Query, cached: EvaluatorResult) -> bool:
        """Whether a stored judgment can be reused instead of re-judging the evaluable.

        A rubric judgment is only meaningful against the rubric it was made against, so editing one
        query's rubric invalidates that query's judgments and nothing else.
        """
        answer = cached.answer
        if not isinstance(answer, RubricJudgment) or answer.rubric_fingerprint is None:
            return True
        return answer.rubric_fingerprint == query.rubric_fingerprint

    def _process_answer(self, llm_response: LLMResponseType[BaseModel], query: Query) -> LLMResponseType[BaseModel]:
        """Processes the raw answer returned by the LLM. Should be implemented by the subclass if needed."""
        return llm_response
