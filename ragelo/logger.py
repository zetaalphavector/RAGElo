from __future__ import annotations

import logging

from rich import print as rich_print

logger = logging.getLogger(__name__)


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
