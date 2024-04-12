import os


def get_path(data_path: str, file_path: str) -> str:
    if file_path.startswith("/"):
        return file_path
    return os.path.join(data_path, file_path)
