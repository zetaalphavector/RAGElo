import logging
import re

logger = logging.getLogger(__name__)


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
                self.print(self.__clean_formatting(msg))
        except Exception:
            self.handleError(record)

    def __clean_formatting(self, msg: str) -> str:
        clean_text = re.sub(r"\[.*?\]", "", msg).strip()
        return clean_text
