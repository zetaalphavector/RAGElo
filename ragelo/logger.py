import logging

from rich import print as rich_print

logger = logging.getLogger(__name__)


class CLILogHandler(logging.Handler):
    """A logging handler for the CLI use case"""

    def emit(self, record):
        msg = self.format(record)
        if record.levelno == logging.WARNING:
            rich_print(f"[yellow]{msg}[/yellow]")
        elif record.levelno == logging.DEBUG:
            rich_print(f"[blue bold]{msg}[/blue bold]")
        elif record.levelno == logging.ERROR:
            rich_print(f"[bold red]{msg}[/bold red]")
        else:
            rich_print(msg)
