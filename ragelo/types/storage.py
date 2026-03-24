from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ragelo.types.results import EloTournamentResult, EvaluatorResult

logger = logging.getLogger(__name__)


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for experiment persistence backends."""

    def initialize(self) -> None: ...

    def save_experiment(self, data: dict[str, Any]) -> None: ...

    def load_experiment(self) -> dict[str, Any] | None: ...

    def save_result(self, result: EvaluatorResult | EloTournamentResult) -> None: ...

    def load_results(self) -> Iterator[str]: ...

    def clear_results(self) -> None: ...


class FileStorageBackend:
    """Persists experiment state and results to JSON / JSONL files on disk."""

    def __init__(self, save_path: Path, evaluations_cache_path: Path):
        self.save_path = save_path
        self.evaluations_cache_path = evaluations_cache_path

    @classmethod
    def default(cls, experiment_name: str, output_dir: str = "ragelo_cache") -> FileStorageBackend:
        """Create a FileStorageBackend with conventional paths based on experiment name."""
        base = Path(output_dir)
        return cls(
            save_path=base / f"{experiment_name}.json",
            evaluations_cache_path=base / f"{experiment_name}_results.jsonl",
        )

    def initialize(self) -> None:
        self.save_path.parent.mkdir(exist_ok=True)
        self.save_path.touch()
        self.evaluations_cache_path.parent.mkdir(exist_ok=True)
        self.evaluations_cache_path.touch()

    def save_experiment(self, data: dict[str, Any]) -> None:
        with self.save_path.open("w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load_experiment(self) -> dict[str, Any] | None:
        if not self.save_path.is_file() or self.save_path.stat().st_size == 0:
            return None
        with self.save_path.open() as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Cache file {self.save_path} is not a valid JSON file")

    def save_result(self, result: EvaluatorResult | EloTournamentResult) -> None:
        with open(self.evaluations_cache_path, "a+") as f:
            f.write(result.model_dump_json() + "\n")

    def load_results(self) -> Iterator[str]:
        if not self.evaluations_cache_path.is_file():
            return
        with self.evaluations_cache_path.open() as f:
            yield from f

    def clear_results(self) -> None:
        if self.evaluations_cache_path.is_file():
            with open(self.evaluations_cache_path, "w"):
                pass


class NullStorageBackend:
    """No-op storage backend for consumers that handle their own persistence."""

    def initialize(self) -> None:
        pass

    def save_experiment(self, data: dict[str, Any]) -> None:
        pass

    def load_experiment(self) -> dict[str, Any] | None:
        return None

    def save_result(self, result: EvaluatorResult | EloTournamentResult) -> None:
        pass

    def load_results(self) -> Iterator[str]:
        return iter(())

    def clear_results(self) -> None:
        pass


def build_storage_backend(experiment_name: str, output_file: str | None = None) -> StorageBackend:
    if output_file:
        save_path = Path(output_file)
        evals_path = save_path.with_name(f"{experiment_name}_results.jsonl")
        return FileStorageBackend(save_path, evals_path)
    return FileStorageBackend.default(experiment_name)
