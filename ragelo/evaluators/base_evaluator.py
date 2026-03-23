from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import TYPE_CHECKING, TypeVar

from jinja2 import Template
from pydantic import create_model

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.presenters import render_failed_evaluations
from ragelo.types.configurations import BaseEvaluatorConfig
from ragelo.types.evaluables import Evaluable
from ragelo.types.formats import LLMResponseType
from ragelo.types.query import Query
from ragelo.types.results import EvaluatorResult
from ragelo.utils import call_async_fn, get_pbar, string_to_template

if TYPE_CHECKING:
    from ragelo.types.experiment import Experiment

T_Result = TypeVar("T_Result", bound=EvaluatorResult)


class BaseEvaluator(ABC):
    """
    An abstract class for all evaluators. An evaluator is responsible for evaluating a query and an evaluable
    """

    config: BaseEvaluatorConfig
    system_prompt: Template | None = None
    user_prompt: Template
    evaluable_name: str = "Evaluable"
    result_type: type[EvaluatorResult]

    def __init__(self, config: BaseEvaluatorConfig, llm_provider: BaseLLMProvider):
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

        async def _consume():
            async for _ in self.evaluate_experiment_async(experiment, n_threads):
                pass

        call_async_fn(_consume)

    async def evaluate_experiment_async(
        self,
        experiment: Experiment,
        n_threads: int | None = None,
        on_result: Callable[[tuple[Query, Evaluable], EvaluatorResult], None] | None = None,
    ) -> AsyncGenerator[tuple[tuple[Query, Evaluable], EvaluatorResult], None]:
        """Evaluate all evaluables in the experiment, yielding results as they complete.

        Args:
            experiment: The experiment to evaluate.
            n_threads: Maximum concurrent LLM calls. Defaults to ``config.n_processes``.
            on_result: Optional callback invoked after each successful evaluation is added
                to the experiment. Receives ``(eval_tuple, evaluation)``.

        Yields:
            Tuples of ``(eval_tuple, evaluation)`` for each completed evaluation.
        """
        n_threads = n_threads or self.config.n_processes
        tuples_to_eval = self._get_tuples_to_evaluate(experiment)
        async for eval_tuple, evaluation in self._run_concurrent_evaluations(
            tuples_to_eval, n_threads, f"Evaluating {self.evaluable_name}s"
        ):
            experiment.add_evaluation(
                eval_tuple,
                evaluation,
                exist_ok=True,
                force=self.config.force,
                should_print=self.config.show_results,
            )
            if on_result is not None:
                on_result(eval_tuple, evaluation)
            yield eval_tuple, evaluation

    def evaluate_all_evaluables(self, query: Query, n_threads: int | None = None):
        """Evaluate all evaluables for a single query, useful for incremental workflows.
        Args:
            query: The query whose evaluables should be evaluated.
            n_threads: Maximum concurrent LLM calls. Defaults to ``config.n_processes``.
        """
        n_threads = n_threads or self.config.n_processes
        call_async_fn(self._evaluate_all_evaluables_async, query, n_threads)

    async def _evaluate_all_evaluables_async(self, query: Query, n_threads: int):
        tuples_to_eval = [(query, e) for e in self._get_all_evaluables(query)]
        async for eval_tuple, evaluation in self._run_concurrent_evaluations(
            tuples_to_eval, n_threads, f"Evaluating {self.evaluable_name}s for query {query.qid}"
        ):
            query.add_evaluation(eval_tuple[1], evaluation, exist_ok=True)

    async def _run_concurrent_evaluations(
        self,
        tuples_to_eval: Sequence[tuple[Query, Evaluable]],
        n_threads: int,
        pbar_desc: str,
    ) -> AsyncGenerator[tuple[tuple[Query, Evaluable], EvaluatorResult], None]:
        pbar = get_pbar(
            len(tuples_to_eval),
            self.config.rich_print,
            desc=pbar_desc,
            disable=not getattr(self.config, "use_progress_bar", True),
        )
        awaitables_ended = False
        pending: set[asyncio.Future] = set()
        tuples_iter = iter(tuples_to_eval)
        future_to_tuple: dict[asyncio.Future, tuple[Query, Evaluable]] = {}
        failed = 0
        evaluations = 0
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
                evaluations += 1
                pbar.update()
                if evaluation.exception:
                    failed += 1
                    continue
                yield eval_tuple, evaluation
        pbar.close()
        if self.config.show_results:
            render_failed_evaluations(evaluations, failed, self.config.rich_print)

    @abstractmethod
    def _get_all_evaluables(self, query: Query) -> list[Evaluable]:
        """Returns all evaluables for a given query"""
        raise NotImplementedError

    @abstractmethod
    async def evaluate_async(self, eval_sample: tuple[Query, Evaluable]) -> EvaluatorResult:
        """Evaluate a single query and evaluable asynchronously."""
        raise NotImplementedError

    @abstractmethod
    def _get_tuples_to_evaluate(self, experiment: Experiment) -> Sequence[tuple[Query, Evaluable]]:
        raise NotImplementedError

    def _process_answer(self, llm_response: LLMResponseType[T_Result], query: Query) -> LLMResponseType[T_Result]:
        """Processes the raw answer returned by the LLM. Should be implemented by the subclass if needed."""
        return llm_response
