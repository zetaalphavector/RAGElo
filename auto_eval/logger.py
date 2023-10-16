import logging

from auto_eval import __app_name__

logger = logging.getLogger(__app_name__)


class CLILogHandler(logging.Handler):
    """A logging handler for the CLI use case"""

    def __init__(self):
        super().__init__()
        try:
            import rich

            self.print = rich.print

            self.rich = True
        except ImportError:
            self.rich = False

    def emit(self, record):
        try:
            msg = self.format(record)
            if self.rich:
                if record.levelno == logging.WARNING:
                    self.print(f"[yellow]{msg}[/yellow]")
                elif record.levelno == logging.DEBUG:
                    self.print(f"[blue bold]{msg}[/blue bold]")
                elif record.levelno == logging.ERROR:
                    self.print(f"[bold red]{msg}[/bold red]")
                else:
                    self.print(msg)
            else:
                self.print(msg)
        except Exception:
            self.handleError(record)
