from pathlib import Path

from typer.testing import CliRunner

from ragelo.cli.cli import app

runner = CliRunner()

ENV = {"OPENAI_API_KEY": "test-key"}
QUERIES_PATH = str(Path("tests/data/queries.csv").resolve())
DOCUMENTS_PATH = str(Path("tests/data/documents.csv").resolve())
ANSWERS_PATH = str(Path("tests/data/answers.csv").resolve())


def _cleanup_files(*paths: str | Path):
    for path in paths:
        path = Path(path)
        if path.exists():
            path.unlink()


def test_run_all_cli(mock_llm_provider_factory):
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
        env=ENV,
    )
    stdout = result.stdout

    assert result.exit_code == 0
    assert "Loaded 2 queries from" in stdout and QUERIES_PATH in stdout
    assert "Loaded 4 new documents from" in stdout and DOCUMENTS_PATH in stdout
    assert "Loaded 4 answers from" in stdout and ANSWERS_PATH in stdout
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


def test_run_reasoner_cli(mock_llm_provider_factory):
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
        env=ENV,
    )
    stdout = result.stdout

    assert result.exit_code == 0
    assert "Loaded 2 queries from" in stdout and QUERIES_PATH in stdout
    assert "Loaded 4 new documents from" in stdout and DOCUMENTS_PATH in stdout
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


def test_run_answer_cli(mock_llm_provider_factory):
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
        env=ENV,
    )
    stdout = result.stdout

    assert result.exit_code == 0
    assert "Loaded 2 queries from" in stdout and QUERIES_PATH in stdout
    assert "Loaded 4 new documents from" in stdout and DOCUMENTS_PATH in stdout
    assert "Loaded 4 answers from" in stdout and ANSWERS_PATH in stdout
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
