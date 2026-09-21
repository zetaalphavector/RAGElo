from __future__ import annotations

from pathlib import Path

from pydantic import Field

from ragelo.types.configurations.base_configs import BaseConfig


class BenchmarkDatasetConfig(BaseConfig):
    data_dir: Path | None = Field(
        default=None,
        description="The directory where the dataset is stored. Defaults to benchmarks/data/<dataset>.",
    )
    split: str | None = Field(default=None, description="The split to judge. Defaults to the dataset's first split.")
    n_samples: int | None = Field(
        default=None,
        description="Judge a sample of this many pairs, games or topics, whichever the dataset samples.",
    )
    seed: int = Field(default=42, description="The seed of the sample and of the Elo tournaments.")


class TrecRag24AnswersDatasetConfig(BenchmarkDatasetConfig):
    n_systems: int = Field(
        default=10,
        description="How many systems play: the best, the worst and a random draw of the ranks between them.",
    )
