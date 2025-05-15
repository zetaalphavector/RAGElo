from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence

import jinja2
import rich

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import BaseEvaluatorConfig
from ragelo.types.evaluables import Evaluable
from ragelo.types.experiment import Experiment
from ragelo.types.formats import AnswerFormat, LLMResponseType
from ragelo.types.query import Query
from ragelo.types.results import EvaluatorResult
from ragelo.utils import call_async_fn, get_pbar


class BaseEvaluator(ABC):
    """
    An abstract class for all evaluators. An evaluator is responsible for evaluating a query and an evaluable
    """

    answer_format: AnswerFormat
    evaluation_prompt: jinja2.Template
    evaluable_name: str = "Evaluable"
    system_prompt: jinja2.Template | None = None

    @abstractmethod
    def __init__(
        self,
        config: BaseEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.force = config.force
        self.verbose = config.verbose
        self.rich_print = config.rich_print
        self.n_processes = config.n_processes
        self.llm_provider = llm_provider
        self.answer_format = config.llm_answer_format
        self.evaluation_prompt = config.evaluation_prompt
        self.llm_answer_schema = config.llm_response_schema

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
        n_threads = n_threads or self.n_processes
        call_async_fn(self._evaluate_experiment_async, experiment, n_threads)

    @abstractmethod
    async def evaluate_async(
        self,
        eval_sample: tuple[Query, Evaluable],
    ) -> EvaluatorResult:
        """Evaluate a single query and evaluable asynchronously."""
        raise NotImplementedError

    async def _evaluate_experiment_async(self, experiment: Experiment, n_threads: int = 1):
        tuples_to_eval = self._get_tuples_to_evaluate(experiment)
        pbar = get_pbar(
            len(tuples_to_eval),
            self.rich_print,
            desc=f"Evaluating {self.evaluable_name}s",
        )

        awaitables_ended = False
        pending: set[asyncio.Future] = set()
        aws = map(self.evaluate_async, tuples_to_eval)
        aws = iter(aws)
        failed = 0
        evaluations = 0
        while pending or not awaitables_ended:
            while len(pending) < n_threads and not awaitables_ended:
                try:
                    aw = next(aws)
                except StopIteration:
                    awaitables_ended = True
                else:
                    pending.add(asyncio.ensure_future(aw))
            if not pending:
                break
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            while done:
                evaluation = await done.pop()
                evaluations += 1
                pbar.update()
                if evaluation.exception:
                    failed += 1
                    continue
                experiment.add_evaluation(evaluation, exist_ok=True, force=self.force, should_print=self.verbose)
        pbar.close()
        if self.verbose:
            self._print_failed_evaluations(evaluations, failed)

    @abstractmethod
    def _get_tuples_to_evaluate(self, queries: Experiment) -> Sequence[tuple[Query, Evaluable]]:
        raise NotImplementedError

    @abstractmethod
    def _process_answer(
        self,
        llm_response: LLMResponseType,
        qid: str,
        agent: str | None = None,
    ) -> EvaluatorResult:
        """Processes the raw answer returned by the LLM. Should be implemented by each subclass."""
        raise NotImplementedError

    def _print_failed_evaluations(self, total_evaluations: int, failed_evaluations: int):
        if self.rich_print:
            rich.print("✅ Done!")
            if failed_evaluations > 0:
                rich.print(f"[bold red]Failed evaluations: {failed_evaluations}[/bold red]")
            rich.print(f"[bold green]Total evaluations: {total_evaluations}[/bold green]")
            return
        print("✅ Done!")
        if failed_evaluations > 0:
            print(f"Failed evaluations: {failed_evaluations}")
        print(f"Total evaluations: {total_evaluations}")
