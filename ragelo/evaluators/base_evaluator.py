import asyncio
import string
from abc import ABC, abstractmethod
from collections.abc import Sequence

import rich
from jinja2 import Environment, Template, meta

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import BaseEvaluatorConfig
from ragelo.types.evaluables import Evaluable
from ragelo.types.experiment import Experiment
from ragelo.types.formats import LLMResponseType
from ragelo.types.query import Query
from ragelo.types.results import EvaluatorResult
from ragelo.utils import call_async_fn, get_pbar


class BaseEvaluator(ABC):
    """
    An abstract class for all evaluators. An evaluator is responsible for evaluating a query and an evaluable
    """

    config: BaseEvaluatorConfig
    system_prompt: Template | None = None
    user_prompt: Template
    evaluable_name: str = "Evaluable"

    @abstractmethod
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        config: BaseEvaluatorConfig,
    ):
        raise NotImplementedError

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
            self.config.rich_print,
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
                experiment.add_evaluation(
                    evaluation,
                    exist_ok=True,
                    force=self.config.force,
                    should_print=self.config.verbose,
                )
        pbar.close()
        if self.config.verbose:
            self._print_failed_evaluations(evaluations, failed)

    @abstractmethod
    def _get_tuples_to_evaluate(self, queries: Experiment) -> Sequence[tuple[Query, Evaluable]]:
        raise NotImplementedError

    def _process_answer(self, llm_response: LLMResponseType) -> LLMResponseType:
        """Processes the raw answer returned by the LLM. Should be implemented by the subclass if needed."""
        return llm_response

    @staticmethod
    def _get_fields_from_string(s: str | Template) -> list[str]:
        """Parse a Jinja2 template or Python format string and return fields in it.

        - If a Jinja2 Template (preferred), use Jinja's meta parser.
        - Otherwise, try Jinja parsing on the string; if none found, fallback to Python's Formatter.
        """
        fields: set[str] = set()
        template_source: str | None = None

        # Attempt Jinja2 parsing first
        try:
            if isinstance(s, Template):
                template_source = getattr(s, "_ragelo_source", None)
                if template_source is not None:
                    env = s.environment or Environment()
                    ast = env.parse(template_source)
                    fields.update(meta.find_undeclared_variables(ast))
            else:
                template_source = s
                ast = Environment().parse(template_source)
                fields.update(meta.find_undeclared_variables(ast))
        except Exception:
            # Ignore Jinja parsing errors; we'll fallback below
            pass

        # Fallback to Python's Formatter if nothing found and we have a string source
        if not fields:
            source_for_python = template_source if template_source is not None else (s if isinstance(s, str) else "")
            fields.update(v[1] for v in string.Formatter().parse(source_for_python) if v[1] is not None)

        return list(fields)

    def _print_failed_evaluations(self, total_evaluations: int, failed_evaluations: int):
        if self.config.rich_print:
            rich.print("✅ Done!")
            if failed_evaluations > 0:
                rich.print(f"[bold red]Failed evaluations: {failed_evaluations}[/bold red]")
            rich.print(f"[bold green]Total evaluations: {total_evaluations}[/bold green]")
            return
        print("✅ Done!")
        if failed_evaluations > 0:
            print(f"Failed evaluations: {failed_evaluations}")
        print(f"Total evaluations: {total_evaluations}")
