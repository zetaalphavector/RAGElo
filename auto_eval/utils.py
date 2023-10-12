import os
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data"


def get_default_data_path(
    step_name: str, extension: str, overwrite: bool = False
) -> Path:
    """Generates a default file path for a given step and extension.
    Args:
        step_name: Name of the step to generate the path for.
        extension: File extension to use.
        overwrite: Whether to overwrite the file if it already exists. If false, creates a new file with a number appended to the name.
    """
    path = DATA_PATH / f"{step_name}.{extension}"
    if path.exists() and not overwrite:
        i = 1
        while path.exists():
            path = DATA_PATH / f"{step_name}_{i}.{extension}"
            i += 1
    return path
