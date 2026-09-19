"""Evaluations per second of a benchmark run, kept beside the experiment so cached re-runs still report it."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

# A retry pass judges a handful of items and mostly measures their timeouts.
MIN_EVALUATIONS = 100


@dataclass(frozen=True, slots=True)
class Run:
    evaluations: int
    seconds: float
    n_processes: int
    calls_per_evaluation: int = 1

    @property
    def per_second(self) -> float:
        return self.evaluations / self.seconds


@dataclass(frozen=True, slots=True)
class Throughput:
    """Every run that judged enough items to time. Single runs vary several-fold with the gateway's
    load, so the rate reported is the median over the runs at the most recent parallelism."""

    runs: tuple[Run, ...] = ()

    def cell(self) -> str:
        if not self.runs:
            return "-"
        last = self.runs[-1]
        rates = [run.per_second for run in self.runs if run.n_processes == last.n_processes]
        parallel = f"x{last.n_processes}"
        if last.calls_per_evaluation > 1:
            parallel += f", {last.calls_per_evaluation} calls each"
        spread = f" [{min(rates):.1f}-{max(rates):.1f}]" if len(rates) > 1 else ""
        return f"{statistics.median(rates):.1f}{spread} ({parallel}, {len(rates)} run{'s' * (len(rates) > 1)})"


def record(path: Path, run: Run) -> Throughput:
    """Adds `run` to the runs in `path` when it judged enough items to time."""
    runs = [Run(**stored) for stored in json.loads(path.read_text())] if path.is_file() else []
    if run.evaluations >= MIN_EVALUATIONS and run.seconds > 0:
        runs.append(run)
        path.write_text(json.dumps([asdict(stored) for stored in runs]))
    return Throughput(tuple(runs))
