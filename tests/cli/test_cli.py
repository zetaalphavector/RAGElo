import csv
import os

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
    # Make sure that the output file was created
    with open("tests/data/reasonings.csv", "r") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["qid", "did", "raw_answer", "answer"]
        assert len(list(reader)) == 4


def test_run_answer_cli():
    result = runner.invoke(
        app,
        [
            "answer-evaluator",
            "pairwise-reasoning",
            "queries.csv",
            "documents.csv",
            "answers.csv",
            "--games-evaluations-file",
            "pairwise_answers_evaluations.csv",
            "--verbose",
            "--data-dir",
            "tests/data/",
        ],
    )
    assert result.exit_code == 0
    assert "✅ Done!" in result.stdout
    assert "Failed evaluations: 0" in result.stdout
    assert "Total evaluations: 2" in result.stdout
    # Make sure that the output file was created
    with open("tests/data/pairwise_answers_evaluations.csv", "r") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "qid",
            "agent_a",
            "agent_b",
            "raw_answer",
            "answer",
            "relevance",
        ]
        assert len(list(reader)) == 2


def test_run_agents_ranker_cli():
    # remove the output file if it exists
    try:
        os.remove("tests/data/agents_ranking.csv")
    except FileNotFoundError:
        pass
    result = runner.invoke(
        app,
        [
            "agents-ranker",
            "elo",
            "pairwise_answers_evaluations.csv",
            "--agents-evaluations-file",
            "agents_ranking.csv",
            "--verbose",
            "--data-dir",
            "tests/data/",
        ],
    )
    assert result.exit_code == 0
    lines = result.stdout.split("\n")
    assert "Agent Scores by elo" in lines[0]
    assert "agent1" in lines[1]
    assert "agent2" in lines[2]

    # Make sure that the output file was created
    with open("tests/data/agents_ranking.csv", "r") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["agent", "score"]
        assert len(list(reader)) == 2
