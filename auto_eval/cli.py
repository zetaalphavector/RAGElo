from typing import Optional

import typer
from loguru import logger
from rich import print
from typing_extensions import Annotated

from auto_eval import __app_name__, __version__
from auto_eval.answer_evaluators import AnswerEvaluatorFactory
from auto_eval.doc_evaluators import DocumentEvaluatorFactory
from auto_eval.elo_ranker import TournamentEloRating
from auto_eval.games_creator import GamesCreator
from auto_eval.games_runner import GamesRunner

app = typer.Typer()


@app.command()
def annotate_documents(
    queries_file: Annotated[
        str,
        typer.Argument(
            help="csv file with queries to evaluate. Each row should have query_id, query"
        ),
    ],
    documents_file: Annotated[
        str,
        typer.Argument(
            help="csv file with documents to evaluate"
            "Each row should have query_id, doc_id, passage"
        ),
    ],
    evaluator_name: Annotated[
        str, typer.Argument(help="Name of the evaluator to use.")
    ] = "reasoner",
    output_file: Annotated[
        Optional[str], typer.Argument(help="csv file to write LLM reasonings to")
    ] = "data/reasonings.csv",
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
    doc_evaluator = DocumentEvaluatorFactory.create(
        evaluator_name,
        query_path=queries_file,
        documents_path=documents_file,
        output_file=output_file,
        model_name=model_name,
        credentials_file=credentials_file,
        print_answers=print_answers,
        force=force,
    )

    doc_evaluator.get_answers()


@app.command()
def annotate_answers(
    queries_file: Annotated[
        str,
        typer.Argument(
            help="csv file with queries to evaluate. Each row should have query_id, query"
        ),
    ],
    answers_file: Annotated[
        str, typer.Argument(help="csv file with the answers each model gave")
    ],
    reasonings_file: Annotated[
        str,
        typer.Argument(
            help="csv file with reasonings for relevancy for each document, "
            "Produced by the annotate-documents command"
        ),
    ],
    evaluator_name: Annotated[
        str, typer.Argument(help="Name of the evaluator to use.")
    ] = "elo",
    output_file: Annotated[
        Optional[str], typer.Argument(help="json file to write pairwise annotators to")
    ] = "data/answers_eval.json",
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
    k: Annotated[int, typer.Argument(help="Number of games to generate.")] = 100,
    bidirectional: Annotated[
        bool, typer.Option("--bidirectional", help="Create games in both directions?")
    ] = False,
):
    """
    Evaluate a list of answers for relevancy using a LLM as annotator
    """
    answer_evaluator = AnswerEvaluatorFactory.create(
        evaluator_name,
        query_path=queries_file,
        answers_file=answers_file,
        reasonings_file=reasonings_file,
        output_file=output_file,
        model_name=model_name,
        credentials_file=credentials_file,
        print_answers=print_answers,
        force=force,
        k=k,
        bidirectional=bidirectional,
    )
    answer_evaluator.prepare()
    answer_evaluator.run()
    answer_evaluator.evaluate()
    answer_evaluator.print_ranking()


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
