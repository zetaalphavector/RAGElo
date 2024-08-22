from typer.testing import CliRunner

from ragelo.cli.cli import app

runner = CliRunner()


def test_app():
    result = runner.invoke(app, ["run-all"])
    assert result.exit_code == 0
    assert "Run all the commands" in result.stdout
