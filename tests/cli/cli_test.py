import csv
import os

from typer.testing import CliRunner

from ragelo.cli.cli import app

runner = CliRunner()

result = runner.invoke(
    app,
    [
        "run-all",
        "queries.csv",
        "documents.csv",
        "answers.csv",
        "--verbose",
        "--data-dir",
        "tests/data/",
    ],
)
assert result.exit_code == 0
assert "Agent Scores by elo" in result.stdout
