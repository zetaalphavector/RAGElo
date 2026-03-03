import logging

from rich.console import Console
from rich.logging import RichHandler

logging.getLogger("ragelo").addHandler(logging.NullHandler())


def configure_logging(level: int | str = "WARNING", *, rich: bool = True) -> None:
    """Configure the ``ragelo`` logger.

    Args:
        level: Logging level (int or name). Examples: ``"INFO"``, ``"DEBUG"``, ``logging.INFO``.
        rich: If True, use RichHandler for colored output; otherwise plain StreamHandler.
    """
    if isinstance(level, str):
        numeric_level = logging.getLevelName(level.upper())
        level = numeric_level if isinstance(numeric_level, int) else logging.WARNING

    lib_logger = logging.getLogger("ragelo")
    lib_logger.setLevel(level)

    lib_logger.handlers = [h for h in lib_logger.handlers if isinstance(h, logging.NullHandler)]

    if rich:
        console = Console()
        handler: logging.Handler = RichHandler(console=console, rich_tracebacks=True, show_time=False, show_path=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    lib_logger.addHandler(handler)
