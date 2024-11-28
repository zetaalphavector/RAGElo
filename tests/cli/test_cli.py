from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from ragelo.cli.cli import app

runner = CliRunner()


@pytest.mark.requires_openai
def test_run_all_cli():
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
            "--experiment-name",
            "test",
            "--output-file",
            "test-output.json",
            "--no-save-results",
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert "Agents Elo Ratings" in result.stdout
    assert os.path.exists("test-output.json")
    os.remove("test-output.json")


@pytest.mark.requires_openai
def test_run_reasoner_cli():
    result = runner.invoke(
        app,
        [
            "retrieval-evaluator",
            "reasoner",
            "queries.csv",
            "documents.csv",
            "--verbose",
            "--data-dir",
            "tests/data/",
            "--experiment-name",
            "test",
            "--output-file",
            "test-output.json",
            "--no-save-results",
            "--force",
        ],
    )
    assert result.exit_code == 0
    assert "✅ Done!" in result.stdout
    assert result.stdout.startswith("🔎 Query ID: 0\n📜 Document ID: 0")
    assert "Total evaluations: 4" in result.stdout
    assert os.path.exists("test-output.json")
    os.remove("test-output.json")


@pytest.mark.requires_openai
def test_run_answer_cli():
    result = runner.invoke(
        app,
        [
            "answer-evaluator",
            "pairwise",
            "queries.csv",
            "documents.csv",
            "answers.csv",
            "--verbose",
            "--data-dir",
            "tests/data/",
            "--experiment-name",
            "test",
            "--output-file",
            "test-output.json",
            "--no-save-results",
            "--force",
            "--bidirectional",
            "--add-reasoning",
        ],
    )
    assert result.exit_code == 0
    assert len(result.stdout.split("✅ Done!")) == 3
    assert len(result.stdout.split("Total evaluations: 4")) == 3
    assert result.stdout.startswith("🔎 Query ID: 0\n📜 Document ID: 0")
    assert "Evaluating Retrieved documents" in result.stdout
    assert "🔎 Query ID: 0\n agent2              🆚   agent1\nParsed Answer: B" in result.stdout
    assert "🔎 Query ID: 1\n agent1              🆚   agent2\nParsed Answer: A" in result.stdout
    assert os.path.exists("test-output.json")
    os.remove("test-output.json")
