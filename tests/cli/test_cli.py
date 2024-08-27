from typer.testing import CliRunner

from ragelo.cli.cli import app

runner = CliRunner()


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
        ],
    )
    assert result.exit_code == 0
    assert "Agent Scores by elo" in result.stdout


def test_run_reasoner_cli():
    result = runner.invoke(
        app,
        [
            "retrieval-evaluator",
            "reasoner",
            "queries.csv",
            "documents.csv",
            "output.csv",
            "--verbose",
            "--data-dir",
            "tests/data/",
        ],
    )
    assert result.exit_code == 0
    assert "✅ Done!" in result.stdout
    assert "Failed evaluations: 0" in result.stdout
    assert "Total evaluations: 4" in result.stdout


def test_run_answer_cli():
    result = runner.invoke(
        app,
        [
            "answer-evaluator",
            "pairwise-reasoning",
            "queries.csv",
            "documents.csv",
            "answers.csv",
            "--document-evaluations-file",
            "reasonings.csv",
            "--verbose",
            "--data-dir",
            "tests/data/",
        ],
    )
    print(result.stdout)
    assert result.exit_code == 0
    assert "✅ Done!" in result.stdout
    assert "Failed evaluations: 0" in result.stdout
    assert "Total evaluations: 2" in result.stdout
