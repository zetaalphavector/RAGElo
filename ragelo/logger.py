import logging

from rich.logging import RichHandler

logger = logging.getLogger("ragelo")

if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def configure_logging(level: int | str = "WARNING", *, rich: bool = True) -> None:
    """Configure ragelo logging.

    Args:
        level: Logging level (int or name). Examples: "INFO", "DEBUG", logging.INFO.
        rich: If True, use RichHandler; otherwise plain text StreamHandler.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if isinstance(level, str):
            level = logging.WARNING

    lib_logger = logging.getLogger("ragelo")
    lib_logger.setLevel(level)

    lib_logger.handlers = []
    lib_logger.propagate = False
    if rich:
        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_time=False, show_path=False)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
    lib_logger.addHandler(handler)
