from typing import Optional

import typer
from typing_extensions import Annotated

from auto_eval import __app_name__, __version__, logger
from auto_eval.answer_evaluators import AnswerEvaluatorFactory
from auto_eval.doc_evaluators import DocumentEvaluatorFactory
from auto_eval.logger import CLILogHandler

logger.addHandler(CLILogHandler())

app = typer.Typer()
state = {
    "verbose": False,
    "force": False,
    "credentials_file": None,
    "model_name": "gpt-4",
}


@app.command()
def annotate_documents(
    queries_file: Annotated[
        str,
        typer.Argument(
            help="csv file with queries to evaluate. Rows should have query_id,query"
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
):
    """
    Evaluate a list of documents for relevancy using a LLM as annotator
    """
    doc_evaluator = DocumentEvaluatorFactory.create(
        evaluator_name,
        query_path=queries_file,
        documents_path=documents_file,
        output_file=output_file,
        model_name=state["model_name"],
        credentials_file=state["credentials_file"],
        print_answers=state["verbose"],
        force=state["force"],
    )

    doc_evaluator.get_answers()


@app.command()
def annotate_answers(
    queries_file: Annotated[
        str,
        typer.Argument(
            help="csv file with queries to evaluate."
            "Each row should have query_id, query"
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


@app.command()
def run_all(
    queries_file: Annotated[
        str,
        typer.Argument(
            help="csv file with queries to evaluate. "
            "Each row should have query_id, query"
        ),
    ],
    documents_file: Annotated[
        str,
        typer.Argument(
            help="csv file with documents to evaluate"
            "Each row should have query_id, doc_id, passage"
        ),
    ],
    answers_file: Annotated[
        str, typer.Argument(help="csv file with the answers each model gave")
    ],
    document_evaluator_name: Annotated[
        str, typer.Argument(help="Name of the document evaluator to use.")
    ] = "reasoner",
    answer_evaluator_name: Annotated[
        str, typer.Argument(help="Name of the answer evaluator to use.")
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
):
    reasoning_file = "data/reasonings.csv"
    doc_evaluator = DocumentEvaluatorFactory.create(
        document_evaluator_name,
        query_path=queries_file,
        documents_path=documents_file,
        output_file=reasoning_file,
        model_name=model_name,
        credentials_file=credentials_file,
        print_answers=print_answers,
        force=force,
    )

    doc_evaluator.get_answers()
    answer_evaluator = AnswerEvaluatorFactory.create(
        answer_evaluator_name,
        query_path=queries_file,
        answers_file=answers_file,
        reasonings_file=reasoning_file,
        output_file=output_file,
        model_name=model_name,
        credentials_file=credentials_file,
        print_answers=print_answers,
        force=force,
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
    verbose: Optional[bool] = typer.Option(
        None,
        "--verbose",
        "-V",
        help="Show debug information",
    ),
    force: Optional[bool] = typer.Option(
        None,
        "--force",
        "-f",
        help="Overwrite output files",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the CLI version",
        callback=_version_callback,
        is_eager=True,
    ),
    credentials_file: Optional[str] = typer.Option(
        None,
        "--credentials",
        "-c",
        help="Path to a file with OpenAI credentials."
        "If missing, will use environment variables",
    ),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use as annotator",
    ),
) -> None:
    """A CLI for auto-eval"""
    if force:
        state["force"] = force
    if verbose:
        state["verbose"] = verbose
    if credentials_file:
        state["credentials_file"] = credentials_file
    if model_name:
        state["model_name"] = model_name
