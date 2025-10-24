import logging
import os

from rich.console import Console
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
        # In tests/CI, avoid colors and wrapping for stable output
        under_pytest = "PYTEST_CURRENT_TEST" in os.environ
        in_ci = os.environ.get("CI") in {"true", "1", "True"}
        force_no_color = os.environ.get("RAGELO_RICH_NO_COLOR") in {"1", "true", "True"}
        no_color = under_pytest or in_ci or force_no_color
        width = 300 if no_color else None
        console = Console(no_color=no_color, width=width)
        handler: logging.Handler = RichHandler(console=console, rich_tracebacks=True, show_time=False, show_path=False)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(levelname)s: %(message)s")
        handler.setFormatter(formatter)
    lib_logger.addHandler(handler)
