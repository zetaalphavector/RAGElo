from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
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
from ragelo.utils import call_async_fn, get_pbar

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
        if config.system_prompt:
            self.system_prompt = config.system_prompt
        if config.user_prompt:
            self.user_prompt = config.user_prompt

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
        call_async_fn(self._evaluate_experiment_async, experiment, n_threads)

    @abstractmethod
    async def evaluate_async(self, eval_sample: tuple[Query, Evaluable]) -> EvaluatorResult:
        """Evaluate a single query and evaluable asynchronously."""
        raise NotImplementedError

    async def _evaluate_experiment_async(self, experiment: Experiment, n_threads: int = 1):
        tuples_to_eval = self._get_tuples_to_evaluate(experiment)
        pbar = get_pbar(
            len(tuples_to_eval),
            self.config.rich_print,
            desc=f"Evaluating {self.evaluable_name}s",
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
                experiment.add_evaluation(
                    eval_tuple,
                    evaluation,
                    exist_ok=True,
                    force=self.config.force,
                    should_print=self.config.verbose,
                )
        pbar.close()
        if self.config.verbose:
            render_failed_evaluations(evaluations, failed, self.config.rich_print)

    @abstractmethod
    def _get_tuples_to_evaluate(self, queries: Experiment) -> Sequence[tuple[Query, Evaluable]]:
        raise NotImplementedError

    def _process_answer(self, llm_response: LLMResponseType[T_Result]) -> LLMResponseType[T_Result]:
        """Processes the raw answer returned by the LLM. Should be implemented by the subclass if needed."""
        return llm_response
