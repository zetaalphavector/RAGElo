from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragelo.cli.cli import app

runner = CliRunner()


def _cleanup_files(*paths: str | Path):
    for path in paths:
        path = Path(path)
        if path.exists():
            path.unlink()


@pytest.mark.requires_openai
def test_run_all_cli():
    experiment_name = "cli-run-all"
    output_file = Path("tests/data/cli-run-all.json")
    results_file = Path("tests/data/cli-run-all_results.jsonl")

    _cleanup_files(output_file, results_file)

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
            experiment_name,
            "--output-file",
            output_file.name,
            "--force",
        ],
    )
    stdout = result.stdout
    # Normalize stdout by removing newlines that Rich may insert for line wrapping
    stdout_normalized = stdout.replace("\n", " ")

    assert result.exit_code == 0
    assert f"Loaded 2 queries from {Path('tests/data/queries.csv').resolve()}" in stdout_normalized
    assert f"Loaded 4 new documents from {Path('tests/data/documents.csv').resolve()}" in stdout_normalized
    assert f"Loaded 4 answers from {Path('tests/data/answers.csv').resolve()}" in stdout_normalized
    assert "Evaluating Retrieved documents" in stdout
    assert "Evaluating Agent Answers" in stdout
    assert "🔎 Query ID: 0" in stdout
    assert "📜 Document ID: 0" in stdout
    assert "👤 Agent A" in stdout
    assert "👤 Agent B" in stdout
    assert "Parsed Answer" in stdout
    assert "Agents Elo Ratings" in stdout
    assert stdout.count("✅ Done!") == 2
    assert stdout.count("Total evaluations: 4") == 1
    assert stdout.count("Total evaluations: 2") == 1

    assert output_file.exists()
    assert results_file.exists()
    assert not Path(f"ragelo_cache/{experiment_name}.json").exists()

    _cleanup_files(output_file, results_file)


@pytest.mark.requires_openai
def test_run_reasoner_cli():
    experiment_name = "cli-reasoner"
    output_file = Path("tests/data/cli-reasoner.json")
    results_file = Path("tests/data/cli-reasoner_results.jsonl")

    _cleanup_files(output_file, results_file)

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
            experiment_name,
            "--output-file",
            output_file.name,
            "--force",
        ],
    )
    stdout = result.stdout
    # Normalize stdout by removing newlines that Rich may insert for line wrapping
    stdout_normalized = stdout.replace("\n", " ")

    assert result.exit_code == 0
    assert f"Loaded 2 queries from {Path('tests/data/queries.csv').resolve()}" in stdout_normalized
    assert f"Loaded 4 new documents from {Path('tests/data/documents.csv').resolve()}" in stdout_normalized
    assert "Evaluating Retrieved documents" in stdout
    assert "🔎 Query ID: 0" in stdout
    assert "📜 Document ID: 0" in stdout
    assert "Parsed Answer" in stdout
    assert stdout.count("✅ Done!") == 1
    assert stdout.count("Total evaluations: 4") == 1

    assert output_file.exists()
    assert results_file.exists()
    assert not Path(f"ragelo_cache/{experiment_name}.json").exists()

    _cleanup_files(output_file, results_file)


@pytest.mark.requires_openai
def test_run_answer_cli():
    experiment_name = "cli-pairwise"
    output_file = Path("tests/data/cli-pairwise.json")
    results_file = Path("tests/data/cli-pairwise_results.jsonl")

    _cleanup_files(output_file, results_file)

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
            experiment_name,
            "--output-file",
            output_file.name,
            "--force",
            "--add-reasoning",
        ],
    )
    stdout = result.stdout
    # Normalize stdout by removing newlines that Rich may insert for line wrapping
    stdout_normalized = stdout.replace("\n", " ")

    assert result.exit_code == 0
    assert f"Loaded 2 queries from {Path('tests/data/queries.csv').resolve()}" in stdout_normalized
    assert f"Loaded 4 new documents from {Path('tests/data/documents.csv').resolve()}" in stdout_normalized
    assert f"Loaded 4 answers from {Path('tests/data/answers.csv').resolve()}" in stdout_normalized
    assert "Evaluating Retrieved documents" in stdout
    assert "Evaluating Agent Answers" in stdout
    assert "🔎 Query ID: 0" in stdout
    assert "📜 Document ID: 0" in stdout
    assert "👤 Agent A" in stdout
    assert "👤 Agent B" in stdout
    assert "Parsed Answer" in stdout
    assert stdout.count("✅ Done!") == 2
    assert stdout.count("Total evaluations: 4") == 1
    assert stdout.count("Total evaluations: 2") == 1

    assert output_file.exists()
    assert results_file.exists()
    assert not Path(f"ragelo_cache/{experiment_name}.json").exists()

    _cleanup_files(output_file, results_file)
