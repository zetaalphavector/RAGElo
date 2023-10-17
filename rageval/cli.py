import os
from typing import Optional

import typer
from typing_extensions import Annotated

from rageval import __app_name__, __version__
from rageval.answer_evaluators import AnswerEvaluatorFactory
from rageval.answer_rankers import AnswerRankerFactory
from rageval.doc_evaluators import DocumentEvaluatorFactory
from rageval.logger import CLILogHandler, logger

logger.addHandler(CLILogHandler())
logger.setLevel("INFO")

app = typer.Typer()
state = {
    "verbose": False,
    "force": False,
    "credentials_file": None,
    "model_name": "gpt-4",
    "data_path": "data",
}


@app.command()
def annotate_documents(
    queries_file: Annotated[
        Optional[str],
        typer.Argument(
            help="csv file with queries to evaluate. Rows should have query_id,query"
        ),
    ] = None,
    documents_file: Annotated[
        Optional[str],
        typer.Argument(
            help="csv file with documents to evaluate"
            "Each row should have query_id, doc_id, passage"
        ),
    ] = None,
    evaluator_name: Annotated[
        str, typer.Argument(help="Name of the evaluator to use.")
    ] = "reasoner",
    output_file: Annotated[
        Optional[str], typer.Option(help="csv file to write LLM reasonings to")
    ] = None,
):
    """
    Evaluate a list of documents for relevancy using a LLM as annotator
    """
    if not queries_file:
        queries_file = os.path.join(state["data_path"], "queries.csv")
        logger.info(f"Using default queries file: {queries_file}")
    if not documents_file:
        documents_file = os.path.join(state["data_path"], "documents.csv")
        logger.info(f"Using default documents file: {documents_file}")
    if not output_file:
        output_file = os.path.join(state["data_path"], "reasonings.csv")
        logger.info(f"Using default output file: {output_file}")

    doc_evaluator = DocumentEvaluatorFactory.create(
        evaluator_name,
        query_path=queries_file,
        documents_path=documents_file,
        output_file=output_file,
        model_name=state["model_name"],
        credentials_file=state["credentials_file"],
        verbose=state["verbose"],
        force=state["force"],
    )

    doc_evaluator.get_answers()


@app.command()
def annotate_answers(
    queries_file: Annotated[
        Optional[str],
        typer.Argument(
            help="csv file with queries to evaluate."
            "Each row should have query_id, query"
        ),
    ] = None,
    answers_file: Annotated[
        Optional[str], typer.Argument(help="csv file with the answers each model gave")
    ] = None,
    reasonings_file: Annotated[
        Optional[str],
        typer.Argument(
            help="csv file with reasonings for relevancy for each document, "
            "Produced by the annotate-documents command"
        ),
    ] = None,
    output_file: Annotated[
        Optional[str],
        typer.Argument(help="json file to write pairwise annotators to"),
    ] = None,
    evaluator_name: Annotated[
        str,
        typer.Argument(help="Name of the evaluator to use."),
    ] = "PairwiseWithReasoning",
    k: Annotated[
        int,
        typer.Option("--k", help="Number of games to generate."),
    ] = 100,
    bidirectional: Annotated[
        bool,
        typer.Option("--bidirectional", help="Create games in both directions?"),
    ] = False,
):
    """
    Evaluate a list of answers for relevancy using a LLM as annotator
    """
    if not queries_file:
        queries_file = os.path.join(state["data_path"], "queries.csv")
        logger.info(f"Using default queries file: {queries_file}")
    if not answers_file:
        answers_file = os.path.join(state["data_path"], "answers.csv")
        logger.info(f"Using default answers file: {answers_file}")
    if not reasonings_file:
        reasonings_file = os.path.join(state["data_path"], "reasonings.csv")
        logger.info(f"Using default reasonings file: {reasonings_file}")
    if not output_file:
        output_file = os.path.join(state["data_path"], "answers_eval.jsonl")
        logger.info(f"Using default output file: {output_file}")

    answer_evaluator = AnswerEvaluatorFactory.create(
        evaluator_name,
        query_path=queries_file,
        answers_file=answers_file,
        reasonings_file=reasonings_file,
        output_file=output_file,
        k=k,
        bidirectional=bidirectional,
        model_name=state["model_name"],
        credentials_file=state["credentials_file"],
        print_answers=state["verbose"],
        force=state["force"],
    )
    answer_evaluator.run()


@app.command()
def rank_agents(
    evaluations_file: Annotated[
        Optional[str],
        typer.Argument(
            help="jsonl file with the annotated answers from annotated_answers"
        ),
    ] = None,
    output_file: Annotated[
        Optional[str],
        typer.Argument(help="csv file to rank to"),
    ] = None,
    evaluator_name: Annotated[
        str,
        typer.Argument(help="Name of the evaluator to use."),
    ] = "elo",
    initial_score: Annotated[
        int,
        typer.Argument(help="Initial score for the Elo ranker"),
    ] = 1000,
    k: Annotated[int, typer.Argument(help="K factor for the Elo ranker")] = 32,
):
    """Ranks answers of agents using an Elo ranker"""
    if not evaluations_file:
        evaluations_file = os.path.join(state["data_path"], "answers_eval.jsonl")
        logger.info(f"Using default evaluations file: {evaluations_file}")
    if not output_file:
        output_file = os.path.join(state["data_path"], "agents_ranking.csv")
        logger.info(f"Using default output file: {output_file}")

    agent_ranker = AnswerRankerFactory.create(
        name=evaluator_name,
        output_file=output_file,
        evaluations_file=evaluations_file,
        initial_score=initial_score,
        k=k,
        print=state["verbose"],
        force=state["force"],
    )
    agent_ranker.evaluate()
    agent_ranker.print_ranking()


@app.command()
def run_all(
    queries_file: Annotated[
        Optional[str],
        typer.Argument(
            help="csv file with queries to evaluate. "
            "Each row should have query_id, query"
        ),
    ] = None,
    documents_file: Annotated[
        Optional[str],
        typer.Argument(
            help="csv file with documents to evaluate"
            "Each row should have query_id, doc_id, passage"
        ),
    ] = None,
    answers_file: Annotated[
        Optional[str],
        typer.Argument(help="csv file with the answers each model gave"),
    ] = None,
    document_evaluator_name: Annotated[
        str, typer.Argument(help="Name of the document evaluator to use.")
    ] = "reasoner",
    answer_evaluator_name: Annotated[
        str, typer.Argument(help="Name of the answer evaluator to use.")
    ] = "PairwiseWithReasoning",
    answer_ranker_name: Annotated[
        str, typer.Argument(help="Name of the answer ranker to use.")
    ] = "elo",
    output_file: Annotated[
        Optional[str], typer.Argument(help="csv file to write ranker result to")
    ] = None,
    k: Annotated[int, typer.Option("--k", help="Number of games to generate.")] = 100,
    bidirectional: Annotated[
        bool, typer.Option("--bidirectional", help="Create games in both directions?")
    ] = False,
    initial_score: Annotated[
        int,
        typer.Argument(help="Initial score for the Elo ranker"),
    ] = 1000,
    elo_k: Annotated[int, typer.Argument(help="K factor for the Elo ranker")] = 32,
):
    if not queries_file:
        queries_file = os.path.join(state["data_path"], "queries.csv")
        logger.info(f"Using default queries file: {queries_file}")
    if not documents_file:
        documents_file = os.path.join(state["data_path"], "documents.csv")
        logger.info(f"Using default documents file: {documents_file}")
    if not answers_file:
        answers_file = os.path.join(state["data_path"], "answers.csv")
        logger.info(f"Using default answers file: {answers_file}")
    if not output_file:
        output_file = os.path.join(state["data_path"], "agents_ranking.csv")
        logger.info(f"Using default output file: {output_file}")

    evaluations_file = os.path.join(state["data_path"], "answers_eval.jsonl")
    reasonings_file = os.path.join(state["data_path"], "reasonings.csv")

    doc_evaluator = DocumentEvaluatorFactory.create(
        document_evaluator_name,
        query_path=queries_file,
        documents_path=documents_file,
        output_file=reasonings_file,
        model_name=state["model_name"],
        credentials_file=state["credentials_file"],
        verbose=state["verbose"],
        force=state["force"],
    )

    doc_evaluator.get_answers()
    answer_evaluator = AnswerEvaluatorFactory.create(
        answer_evaluator_name,
        query_path=queries_file,
        answers_file=answers_file,
        reasonings_file=reasonings_file,
        output_file=evaluations_file,
        k=k,
        bidirectional=bidirectional,
        model_name=state["model_name"],
        credentials_file=state["credentials_file"],
        print_answers=state["verbose"],
        force=state["force"],
    )
    answer_evaluator.run()

    agent_ranker = AnswerRankerFactory.create(
        name=answer_ranker_name,
        output_file=output_file,
        evaluations_file=evaluations_file,
        initial_score=initial_score,
        k=elo_k,
        print=state["verbose"],
        force=state["force"],
    )
    agent_ranker.evaluate()
    agent_ranker.print_ranking()


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
        help="Path to a file with OpenAI credentials. "
        "If missing, will use environment variables",
    ),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model to use as annotator",
    ),
    data_path: str = typer.Option(
        "data",
        "--data-path",
        help="Base path to store output data",
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
    if data_path:
        state["data_path"] = data_path.rstrip("/")
