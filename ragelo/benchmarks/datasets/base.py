from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar, get_type_hints

from rich.console import RenderableType

from ragelo import Experiment
from ragelo.benchmarks.agreement import _SCIPY_AVAILABLE
from ragelo.benchmarks.pricing import Price
from ragelo.benchmarks.throughput import Run, Throughput, record
from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types.configurations import BenchmarkDatasetConfig
from ragelo.types.evaluables import Evaluable
from ragelo.types.types import BenchmarkDatasetTypes
from ragelo.utils import warn_ignored_arguments

T_Evaluator = TypeVar("T_Evaluator", bound=BaseEvaluator)
T_Outcome = TypeVar("T_Outcome")


class Dataset(ABC, Generic[T_Evaluator, T_Outcome]):
    """Human judgments, the experiment an evaluator judges in their place, and the comparison of the two."""

    config: BenchmarkDatasetConfig
    data_name: ClassVar[str]
    splits: ClassVar[tuple[str, ...]] = ("test",)
    calls_per_evaluation: ClassVar[int] = 1

    def __init__(self, config: BenchmarkDatasetConfig):
        if not _SCIPY_AVAILABLE:
            raise ImportError("The benchmarks need scipy. Install it with: pip install 'ragelo[benchmarks]'")
        self.config = config
        self.split = config.split or self.splits[0]
        if self.split not in self.splits:
            raise ValueError(f"{self.data_name} has no {self.split} split. Choose one of {', '.join(self.splits)}.")
        self.data_dir = config.data_dir or Path("benchmarks/data") / self.data_name

    @classmethod
    def get_config_class(cls) -> type[BenchmarkDatasetConfig]:
        return get_type_hints(cls)["config"]

    @abstractmethod
    def download(self) -> None: ...

    @property
    @abstractmethod
    def subset(self) -> str:
        """Names what was sampled, so that every sample has its own cache."""

    @abstractmethod
    def get_evaluator(self, variant: str, llm_provider: BaseLLMProvider, **kwargs: Any) -> T_Evaluator:
        """`kwargs` are options of the evaluator's config."""

    @abstractmethod
    def to_experiment(self, experiment_name: str, save_path: str | None = None) -> Experiment:
        """Nothing an evaluator can render into a prompt may carry the human judgments."""

    @abstractmethod
    def items(self, experiment: Experiment) -> Sequence[Evaluable]: ...

    @abstractmethod
    def outcome(self, experiment: Experiment, evaluator: T_Evaluator, throughput: Throughput) -> T_Outcome: ...

    @abstractmethod
    def tables(
        self, outcomes: Mapping[tuple[str, str], T_Outcome], prices: Mapping[str, Price]
    ) -> list[RenderableType]:
        """`outcomes` and `prices` are keyed by (variant, model) and by model."""

    def judge(
        self,
        evaluator: T_Evaluator,
        experiment_name: str,
        results_dir: Path | str,
        clock: Callable[[], float] = time.perf_counter,
    ) -> T_Outcome:
        """Evaluations are cached per experiment, so a re-run only judges what is missing or failed."""
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        experiment = self.to_experiment(experiment_name, save_path=str(results_dir / f"{experiment_name}.json"))
        name = str(evaluator.config.evaluator_name)
        cached = sum(name in item.evaluations for item in self.items(experiment))
        started = clock()
        evaluator.evaluate_experiment(experiment)
        seconds = clock() - started
        judged = sum(name in item.evaluations for item in self.items(experiment))
        run = Run(judged - cached, seconds, evaluator.config.n_processes, self.calls_per_evaluation)
        return self.outcome(experiment, evaluator, record(results_dir / f"{experiment_name}_throughput.json", run))


class DatasetFactory:
    registry: ClassVar[dict[BenchmarkDatasetTypes, type[Dataset[Any, Any]]]] = {}

    @classmethod
    def register(
        cls, dataset_name: BenchmarkDatasetTypes
    ) -> Callable[[type[Dataset[Any, Any]]], type[Dataset[Any, Any]]]:
        def inner_wrapper(wrapped_class: type[Dataset[Any, Any]]) -> type[Dataset[Any, Any]]:
            cls.registry[dataset_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls, dataset_name: BenchmarkDatasetTypes, config: BenchmarkDatasetConfig | None = None, **kwargs: Any
    ) -> Dataset[Any, Any]:
        if config is None:
            config_class = cls.registry[dataset_name].get_config_class()
            warn_ignored_arguments(f"The {dataset_name} dataset", config_class, kwargs)
            config = config_class(**kwargs)
        return cls.registry[dataset_name](config)


def get_dataset(
    dataset_name: BenchmarkDatasetTypes | str, config: BenchmarkDatasetConfig | None = None, **kwargs: Any
) -> Dataset[Any, Any]:
    try:
        dataset_name = BenchmarkDatasetTypes(dataset_name)
    except ValueError:
        raise ValueError(f"Unknown dataset {dataset_name}")
    return DatasetFactory.create(dataset_name, config=config, **kwargs)
