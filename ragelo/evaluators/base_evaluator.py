from __future__ import annotations

import asyncio
import string
from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from tqdm.auto import tqdm

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
from ragelo.types.configurations import BaseEvaluatorConfig
from ragelo.types.evaluables import Evaluable
from ragelo.types.experiment import Experiment
from ragelo.types.formats import AnswerFormat
from ragelo.types.query import Query
from ragelo.types.results import EvaluatorResult


class BaseEvaluator(ABC):
    """
    An abstract class for all evaluators. An evaluator is responsible for evaluating a query and an evaluable
    """

    config: BaseEvaluatorConfig
    answer_format: AnswerFormat
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

        def run(coroutine):
            return asyncio.run(coroutine)

        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run, self._evaluate_experiment_async(experiment, n_threads))
                _ = future.result()
        except RuntimeError:
            _ = asyncio.run(self._evaluate_experiment_async(experiment, n_threads))

    @abstractmethod
    async def evaluate_async(
        self,
        eval_sample: tuple[Query, Evaluable],
    ) -> EvaluatorResult:
        """Evaluate a single query and evaluable asynchronously."""
        raise NotImplementedError

    async def _evaluate_experiment_async(self, experiment: Experiment, n_threads: int = 1):
        tuples_to_eval = self._get_tuples_to_evaluate(experiment)
        if self.config.rich_print:
            import warnings

            from tqdm import TqdmExperimentalWarning

            warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
            from tqdm.rich import tqdm as rich_tqdm

            pbar_fn = rich_tqdm  # type: ignore
        else:
            pbar_fn = tqdm  # type: ignore

        pbar = pbar_fn(
            total=len(tuples_to_eval),
            ncols=100,
            desc=f"Evaluating {self.evaluable_name}s",
            disable=not self.config.use_progress_bar,
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
                experiment.add_evaluation(evaluation, exist_ok=True)
        pbar.close()
        if self.config.verbose:
            self._print_failed_evaluations(evaluations, failed)

    @abstractmethod
    def _get_tuples_to_evaluate(self, queries: Experiment) -> Sequence[tuple[Query, Evaluable]]:
        raise NotImplementedError

    def _validate_answer(
        self,
        answer: dict[str, Any] | PydanticBaseModel | str,
    ):
        """Ensures that the LLM output is properly formatted."""

        if self.config.llm_answer_format == AnswerFormat.JSON:
            if not isinstance(answer, dict):
                raise ValueError(f"Expected LLM answer as a JSON dictionary, got {type(answer)}: {answer}")
            if self.config.llm_response_schema is not None:
                if isinstance(self.config.llm_response_schema, dict):
                    schema = self.config.llm_response_schema
                    if not all(k in answer for k in schema.keys()):
                        raise ValueError(f"Expected LLM answer to have keys {schema.keys()}, got {answer.keys()}")
        elif self.config.llm_answer_format == AnswerFormat.STRUCTURED:
            if not isinstance(answer, PydanticBaseModel):
                raise ValueError(f"Expected LLM answer as a PydanticBaseModel, got {type(answer)}: {answer}")
        elif self.config.llm_answer_format == AnswerFormat.TEXT:
            if not isinstance(answer, str):
                raise ValueError(f"Expected LLM answer as a string, got {type(answer)}: {answer}")

    def _process_answer(
        self, raw_answer: str | dict[str, Any] | PydanticBaseModel
    ) -> float | str | dict[str, Any] | PydanticBaseModel:
        """Processes the raw answer returned by the LLM. Should be implemented by the subclass if needed."""
        self._validate_answer(raw_answer)
        return raw_answer

    @staticmethod
    def _get_fields_from_string(s: str) -> list[str]:
        """Parse a formatted string and return all the fields in it"""
        field_names = [v[1] for v in string.Formatter().parse(s) if v[1] is not None]
        return field_names

    @staticmethod
    def _get_usable_fields_from_metadata(
        prompt: str, metadata: dict[str, str] | None, skip_fields: list[str] = []
    ) -> dict[str, str]:
        """Get the fields from the prompt that are in the metadata"""
        expected_fields = BaseEvaluator._get_fields_from_string(prompt)
        valid_fields: dict[str, str] = {}
        if metadata is None:
            return valid_fields
        for field in expected_fields:
            if field in metadata and field not in skip_fields:
                valid_fields[field] = metadata[field]
        return valid_fields

    def _print_failed_evaluations(self, total_evaluations: int, failed_evaluations: int):
        if self.config.rich_print:
            try:
                import rich

                rich.print("✅ Done!")
                rich.print(f"Failed evaluations: {failed_evaluations}")
                rich.print(f"Total evaluations: {total_evaluations}")
                return
            except ImportError:
                logger.warning("Rich not installed. Using plain print")
        print("✅ Done!")
        print(f"Failed evaluations: {failed_evaluations}")
        print(f"Total evaluations: {total_evaluations}")
