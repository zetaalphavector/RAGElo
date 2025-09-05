from __future__ import annotations

import os


def get_path(data_path: str | None, file_path: str, check_exists: bool = True) -> str:
    if data_path is None:
        data_path = "."
    if file_path.startswith("/"):
        abs_path = file_path
    abs_path = os.path.abspath(os.path.join(data_path, file_path))
    if check_exists:
        assert os.path.exists(abs_path), f"File {abs_path} does not exist"
    return abs_path
