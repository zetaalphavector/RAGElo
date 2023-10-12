from typing import Optional

import typer
from typing_extensions import Annotated

from auto_eval import __app_name__, __version__
from auto_eval.documents_eval import DocumentEvaluator
from auto_eval.games_creator import GamesCreator
from auto_eval.games_runner import GamesRunner

app = typer.Typer()


@app.command()
def create_relevancy_reasoning(
    documents_path: Annotated[
        str,
        typer.Argument(
            help="csv file with documents to evaluate"
            "Each row should have query_id, Query, doc_id, title, passage"
        ),
    ],
    output_file: Annotated[
        str, typer.Argument(help="csv file to write LLM reasonings to")
    ],
    prompt_name: Annotated[
        str, typer.Option(help="Prompt to use for evaluating documents")
    ] = "prompt_relevancy",
    model_name: Annotated[
        str, typer.Option(help="Model to use as annotator")
    ] = "gpt-4",
    print_answers: Annotated[
        bool, typer.Option("--print", help="Print LLM answers to screen?")
    ] = False,
    credentials_file: Annotated[
        Optional[str],
        typer.Option(
            help="path to a file with OpenAI credentials."
            "If missing, will use environment variables"
        ),
    ] = None,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite output file?")
    ] = False,
):
    """
    Evaluate a list of documents for relevancy using a LLM as annotator
    """
    # If a credentials file was supplied, extract values and set as environment variables
    document_evaluator = DocumentEvaluator(
        documents_path, prompt_name, model_name, credentials_file, print_answers, force
    )

    document_evaluator.get_answers(output_file)


@app.command()
def create_pairwise_games(
    queries_file: Annotated[
        str,
        typer.Argument(
            help="csv file with queries to evaluate"
            "Each row should have query_id, Query"
        ),
    ],
    answers_path: Annotated[
        str,
        typer.Argument(
            help="Directory with JSONL files with answers from the different models"
        ),
    ],
    output_file: Annotated[str, typer.Argument(help="csv file to write games to")],
    reasonings_file: Annotated[
        str,
        typer.Argument(
            help="csv file with LLM reasonings for relevancy for each document"
        ),
    ],
    k: Annotated[int, typer.Argument(help="Number of games to generate")] = 100,
    prompt_template: Annotated[
        str, typer.Option(help="Prompt to use for creating games")
    ] = "pairwise_prompt",
    bidirectional: Annotated[
        bool, typer.Option("--bidirectional", help="Create games in both directions?")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite output file?")
    ] = False,
    print: Annotated[
        bool, typer.Option("--print", help="Print games to screen?")
    ] = False,
):
    """Generates random pairs of agents and build the prompts for an Evaluator to evaluate them."""
    games_creator = GamesCreator(
        queries_file,
        answers_path,
        output_file,
        k,
        reasonings_file,
        prompt_template,
        bidirectional,
        force,
        print,
    )
    games_creator.create_all_prompts()
    pass


@app.command()
def run_games(
    games_file: Annotated[
        str,
        typer.Argument(help="jsonl file with prompts to be used. One per line."),
    ],
    output_file: Annotated[str, typer.Argument(help="csv file to write results to")],
    answers_file: Annotated[str, typer.Argument(help="jsonl file to write answers to")],
    model_name: Annotated[
        str, typer.Argument(help="Model to use as Evaluator")
    ] = "gpt-4",
    credentials_file: Annotated[
        Optional[str],
        typer.Option(
            help="path to a file with OpenAI credentials."
            "If missing, will use environment variables"
        ),
    ] = None,
    print_answers: Annotated[
        bool, typer.Option("--print", help="Print LLM answers to screen?")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Overwrite output file?")
    ] = False,
):
    """Uses an Evaluator to evaluate a series of pairwise games with prompts created by the create-pairwise-games command."""
    runner = GamesRunner(
        games_file, output_file, model_name, credentials_file, print_answers, force
    )
    runner.run_games()


@app.command()
def evaluate_games():
    pass


@app.command()
def run_all():
    pass


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"{__app_name__} v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the CLI version",
        callback=_version_callback,
        is_eager=True,
    )
) -> None:
    """A CLI for auto-eval"""
    return
