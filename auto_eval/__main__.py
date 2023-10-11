from loguru import logger
from rich.console import Console

from auto_eval import __app_name__

from . import cli

console = Console()
logger.remove()


def _log_formatter(record: dict) -> str:
    """Log message formatter"""
    color_map = {
        "TRACE": "dim blue",
        "DEBUG": "blue bold",
        "INFO": "bold",
        "SUCCESS": "bold green",
        "WARNING": "yellow",
        "ERROR": "bold red",
        "CRITICAL": "bold white on red",
    }
    lvl_color = color_map.get(record["level"].name, "cyan")
    lvl_name = record["level"].name
    return (
        "[not bold green]{time:YYYY/MM/DD HH:mm:ss.SSS}[/not bold green] |"
        f" [{lvl_color}] {lvl_name: <8} [/{lvl_color}]| "
        f"[not bold cyan]{record['name']}[/not bold cyan]:"
        f"[not bold cyan]{record['function']}[/not bold cyan]:"
        f"[not bold cyan]{record['line']}[/not bold cyan]"
        f" - [{lvl_color}]{{message}}[/{lvl_color}]"
    )


logger.add(
    console.print,
    format=_log_formatter,  # type: ignore
    level="INFO",
    colorize=True,
)


def main():
    logger.enable("auto_eval")
    cli.app(prog_name=__app_name__)


if __name__ == "__main__":
    main()
