import logging

from rich import print as rich_print

logger = logging.getLogger("ragelo")

if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class CLILogHandler(logging.Handler):
    """A logging handler for the CLI use case"""

    def __init__(self, use_rich: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_rich = use_rich

    def emit(self, record):
        msg = self.format(record)
        if record.levelno == logging.WARNING:
            if self.use_rich:
                rich_print(f"[yellow]{msg}[/yellow]")
            else:
                print("[WARNING] " + msg)
        elif record.levelno == logging.DEBUG:
            if self.use_rich:
                rich_print(f"[blue bold]{msg}[/blue bold]")
            else:
                print("[DEBUG] " + msg)
        elif record.levelno == logging.ERROR:
            if self.use_rich:
                rich_print(f"[bold red]{msg}[/bold red]")
            else:
                print("[ERROR] " + msg)
        else:
            if self.use_rich:
                rich_print(msg)
            else:
                print(msg)


def configure_logging(level: int | str = "WARNING", *, rich: bool = True) -> None:
    """Configure ragelo logging.

    Args:
        level: Logging level (int or name). Examples: "INFO", "DEBUG", logging.INFO.
        rich: If True, use rich-colored output via CLILogHandler; otherwise plain text.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if isinstance(level, str):
            level = logging.WARNING

    lib_logger = logging.getLogger("ragelo")
    lib_logger.setLevel(level)

    lib_logger.handlers = []
    lib_logger.propagate = False
    cli_handler = CLILogHandler(use_rich=rich)
    formatter = logging.Formatter("%(message)s")
    cli_handler.setFormatter(formatter)
    lib_logger.addHandler(cli_handler)
