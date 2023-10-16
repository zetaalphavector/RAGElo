import logging

from auto_eval import __app_name__

logger = logging.getLogger(__app_name__)


class CLILogger(logging.Logger):
    """A fake "logger" for the CLI use case"""

    def __init__(self, name: str = __app_name__):
        super().__init__(name)
        try:
            import rich

            self.print = rich.print

            self.rich = True
        except ImportError:
            self.rich = False

    def warning(self, msg: str):
        lvl_name = "WARNING"
        if self.rich:
            self.print(f"[yellow]{msg}[/yellow]")
        else:
            self.print(f"{lvl_name:<8}| {msg}")

    def debug(self, msg: str):
        lvl_name = "DEBUG"
        if self.rich:
            self.print(f"[blue bold]{msg}[/blue bold]")
        else:
            self.print(f"{lvl_name:<8}| {msg}")

    def info(self, msg: str):
        lvl_name = "INFO"
        if self.rich:
            self.print(msg)
        else:
            self.print(f"{lvl_name:<8}| {msg}")

    def error(self, msg: str):
        lvl_name = "ERROR"
        if self.rich:
            self.print(f"[bold red]{msg}[/bold red]")
        else:
            self.print(f"{lvl_name:<8}| {msg}")
