"""The CLI's parameters are generated from the config classes by `ragelo.cli.args`."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Annotated, Any, get_type_hints

import pytest
import typer
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import PydanticUndefined
from typer.models import ArgumentInfo, OptionInfo
from typer.testing import CliRunner

from ragelo.cli.args import cli_type, config_command, config_parameters
from ragelo.cli.cli import app
from ragelo.types import BaseConfig
from ragelo.types.types import RetrievalEvaluatorTypes


class ToyConfig(BaseConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    queries_csv_file: Annotated[str, typer.Argument()] = Field(default="queries.csv", description="The queries.")
    top_k: int = Field(default=10, description="How many documents.")
    model: str | None = None
    grades: list[str] | None = None
    evaluator_name: str | RetrievalEvaluatorTypes = "reasoner"
    weights: dict[str, float] = Field(default_factory=dict)
    rubrics: dict[str, list[str]] | None = None
    keep: Callable[[str], bool] | None = None
    prompt: Template | str | None = None
    schema_class: type[BaseModel] | None = None


def commands(group: Any, path: tuple[str, ...] = ()) -> dict[str, Any]:
    """Typer vendors Click from 0.27 on and depends on it before, so commands are told apart by shape."""
    if not hasattr(group, "commands"):
        return {" ".join(path): group}
    found: dict[str, Any] = {}
    for name, command in group.commands.items():
        found |= commands(command, (*path, name))
    return found


COMMANDS = commands(typer.main.get_command(app))


def refuses(path: str, arguments: list[str]) -> str:
    """The name of the Click error a command line is refused with."""
    with pytest.raises(Exception) as refusal:
        parse(path, arguments)
    return type(refusal.value).__name__


def parse(path: str, arguments: list[str]) -> dict[str, Any]:
    """What Click parses for a command, without running it."""
    return COMMANDS[path].make_context(path, arguments).params


class TestConfigParameters:
    def test_each_kind_of_field_becomes_the_parameter_the_command_line_can_express(self):
        parameters = {p.name: p for p in config_parameters(ToyConfig) if p.name in ToyConfig.__annotations__}

        assert {name: p.annotation for name, p in parameters.items()} == {
            "queries_csv_file": str,
            "top_k": int,
            "model": str,
            "grades": list[str],
            "evaluator_name": str,
        }, (
            "an optional loses its None, a union takes its first type, and dicts, callables, templates and classes are left out"
        )
        assert isinstance(parameters["queries_csv_file"].default, ArgumentInfo)
        assert all(isinstance(p.default, OptionInfo) for name, p in parameters.items() if name != "queries_csv_file")
        assert (parameters["top_k"].default.default, parameters["top_k"].default.help) == (10, "How many documents.")

    def test_the_command_receives_the_config_built_from_the_command_line(self):
        received: list[ToyConfig] = []
        toy = typer.Typer()

        @toy.command()
        @config_command
        def judge(config: ToyConfig) -> None:
            received.append(config)

        result = CliRunner().invoke(toy, ["q.csv", "--top-k", "3", "--grades", "a", "--grades", "b", "--force"])

        assert result.exit_code == 0, result.output
        [config] = received
        assert isinstance(config, ToyConfig)
        assert (config.queries_csv_file, config.top_k, config.grades, config.force) == ("q.csv", 3, ["a", "b"], True)


class TestCommands:
    @pytest.mark.parametrize("path", sorted(COMMANDS))
    def test_a_command_takes_every_field_of_its_config_that_the_command_line_can_express(self, path):
        hints = get_type_hints(inspect.unwrap(COMMANDS[path].callback))
        [config_class] = [hint for name, hint in hints.items() if name != "return"]
        expected = {name: field for name, field in config_class.model_fields.items() if cli_type(field) is not None}
        parameters = {parameter.name: parameter for parameter in COMMANDS[path].params}

        assert parameters.keys() == expected.keys()
        for name, field in expected.items():
            default = None if field.default is PydanticUndefined else field.default
            assert getattr(parameters[name].default, "value", parameters[name].default) == default, name
            assert parameters[name].help == field.description, name
            is_positional = any(isinstance(marker, ArgumentInfo) for marker in field.metadata)
            assert (parameters[name].param_type_name == "argument") is is_positional, name


class TestParsing:
    def test_positional_files_fall_back_to_their_defaults(self):
        parsed = parse("run-all", ["my_queries.csv"])
        assert (parsed["queries_csv_file"], parsed["documents_csv_file"], parsed["answers_csv_file"]) == (
            "my_queries.csv",
            "documents.csv",
            "answers.csv",
        )

    def test_a_repeated_option_collects_every_value(self):
        parsed = parse(
            "retrieval-evaluator reasoner", ["--relevance-grades", "off topic", "--relevance-grades", "on topic"]
        )
        assert list(parsed["relevance_grades"]) == ["off topic", "on topic"]

    @pytest.mark.parametrize(("flag", "value"), [("--force", True), ("--no-force", False)])
    def test_a_boolean_has_an_on_and_an_off_flag(self, flag, value):
        assert parse("answer-evaluator pairwise", [flag])["force"] is value

    def test_numbers_are_parsed_and_a_wrong_type_is_refused(self):
        assert parse("run-all", ["--n-processes", "4", "--elo-k", "16"])["n_processes"] == 4
        assert refuses("run-all", ["--n-processes", "many"]) == "BadParameter"

    def test_an_optional_value_left_out_is_none(self):
        parsed = parse("retrieval-evaluator rdnam", [])
        assert (parsed["model"], parsed["guidelines"], parsed["output_file"]) == (None, None, None)

    def test_a_choice_only_takes_its_values(self):
        assert (
            parse("answer-evaluator pairwise", ["--evaluator-name", "chat_pairwise"])["evaluator_name"]
            == "chat_pairwise"
        )
        assert refuses("answer-evaluator pairwise", ["--evaluator-name", "best_of_three"]) == "BadParameter"

    @pytest.mark.parametrize(
        "hidden", ["--system-prompt", "--user-prompt", "--llm-response-schema", "--document-filter"]
    )
    def test_prompts_schemas_and_callables_are_not_options(self, hidden):
        assert refuses("answer-evaluator pairwise", [hidden, "x"]) == "NoSuchOption"
