from importlib import metadata

_PYDANTIC_MAJOR_VERSION: int = int(metadata.version("pydantic").split(".")[0])
