from __future__ import annotations

import os
from pathlib import Path

from ragelo.types.storage import FileStorageBackend, StorageBackend


def get_path(data_path: str | None, file_path: str, check_exists: bool = True) -> str:
    if data_path is None:
        data_path = "."
    if file_path.startswith("/"):
        abs_path = file_path
    else:
        abs_path = os.path.abspath(os.path.join(data_path, file_path))
    if check_exists:
        assert os.path.exists(abs_path), f"File {abs_path} does not exist"
    return abs_path


def build_storage_backend(experiment_name: str, output_file: str | None = None) -> StorageBackend:
    if output_file:
        save_path = Path(output_file)
        evals_path = save_path.with_name(f"{experiment_name}_results.jsonl")
        return FileStorageBackend(save_path, evals_path)
    return FileStorageBackend.default(experiment_name)
