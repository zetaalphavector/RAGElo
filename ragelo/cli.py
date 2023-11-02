import os
from dataclasses import dataclass
from typing import Optional

import typer
from typing_extensions import Annotated

from ragelo import __app_name__, __version__
from ragelo.answer_evaluators import AnswerEvaluatorFactory
from ragelo.answer_rankers import AnswerRankerFactory
from ragelo.doc_evaluators import DocumentEvaluatorFactory
from ragelo.logger import CLILogHandler, logger

logger.addHandler(CLILogHandler())
logger.setLevel("INFO")


@dataclass
class State:
    verbose: bool = True
    force: bool = False
    credentials_file: str | None = None
    model_name: str = "gpt-4"
    data_path: str = "data"


app = typer.Typer()


state = State()


@app.command()
def documents_annotator(
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
    output_file: Annotated[
        Optional[str],
        typer.Option(help="csv file to write LLM reasonings to"),
    ] = None,
    evaluator_name: Annotated[
        str,
        typer.Argument(help="Name of the evaluator to use."),
    ] = "reasoner",
):
    """
    Evaluate a list of documents for relevancy using a LLM as annotator
    """
    if not output_file:
        output_file = os.path.join(state.data_path, "reasonings.csv")
        logger.info(f"Using default output file: {output_file}")

    doc_evaluator = DocumentEvaluatorFactory.create(
        evaluator_name,
        query_path=queries_file,
        documents_path=documents_file,
        output_file=output_file,
        model_name=state.model_name,
        credentials_file=state.credentials_file,
        verbose=state.verbose,
        force=state.force,
    )

    doc_evaluator.annotate_all_docs()


@app.command()
def answers_annotator(
    queries_file: Annotated[
        str,
        typer.Argument(
            help="csv file with queries to evaluate."
            "Each row should have query_id, query"
        ),
    ],
    answers_file: Annotated[
        str,
        typer.Argument(help="csv file with the the answers generated by each agent"),
    ],
    reasonings_file: Annotated[
        Optional[str],
        typer.Argument(
            help="csv file with reasonings for relevancy for each document, "
            "Produced by the annotate-documents command. Only needed for the "
            "PairwiseWithReasoning evaluator"
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
        typer.Option("--k", help="Number of pairs to annotate."),
    ] = 100,
    bidirectional: Annotated[
        bool,
        typer.Option("--bidirectional", help="Create games in both directions?"),
    ] = False,
):
    """
    Evaluate a list of answers for relevancy using a LLM as annotator
    """
    if not reasonings_file and evaluator_name == "PairwiseWithReasoning":
        raise ValueError(
            "You need to provide a reasonings file when using PairwiseWithReasoning"
        )
    if not output_file:
        output_file = os.path.join(state.data_path, "answers_eval.jsonl")
        logger.info(f"Using default output file: {output_file}")

    answer_evaluator = AnswerEvaluatorFactory.create(
        evaluator_name,
        query_path=queries_file,
        answers_file=answers_file,
        reasonings_file=reasonings_file,
        output_file=output_file,
        k=k,
        bidirectional=bidirectional,
        model_name=state.model_name,
        credentials_file=state.credentials_file,
        print_answers=state.verbose,
        force=state.force,
    )
    answer_evaluator.evaluate_all_answers()


@app.command()
def agents_ranker(
    evaluations_file: Annotated[
        str,
        typer.Argument(
            help="jsonl file with the annotated answers from annotated_answers"
        ),
    ],
    output_file: Annotated[
        str,
        typer.Argument(help="csv file to rank to"),
    ],
    evaluator_name: Annotated[
        str,
        typer.Option("--ranker", help="Name of the evaluator to use."),
    ] = "elo",
    initial_score: Annotated[
        int,
        typer.Option("--elo_initial_score", help="Initial score for the Elo ranker"),
    ] = 1000,
    k: Annotated[int, typer.Option("--elo_k", help="K factor for the Elo ranker")] = 32,
):
    """Ranks answers of agents using an Elo ranker"""

    if not output_file:
        output_file = os.path.join(state.data_path, "agents_ranking.csv")
        logger.info(f"Using default output file: {output_file}")

    agent_ranker = AnswerRankerFactory.create(
        name=evaluator_name,
        output_file=output_file,
        evaluations_file=evaluations_file,
        initial_score=initial_score,
        k=k,
        print=state.verbose,
        force=state.force,
    )
    agent_ranker.evaluate()
    agent_ranker.print_ranking()


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
        str,
        typer.Argument(help="csv file with the answers each model gave"),
    ],
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
        typer.Option("--elo_initial_score", help="Initial score for the Elo ranker"),
    ] = 1000,
    elo_k: Annotated[
        int, typer.Option("--elo_k", help="K factor for the Elo ranker")
    ] = 32,
):
    if not queries_file:
        queries_file = os.path.join(state.data_path, "queries.csv")
        logger.info(f"Using default queries file: {queries_file}")
    if not documents_file:
        documents_file = os.path.join(state.data_path, "documents.csv")
        logger.info(f"Using default documents file: {documents_file}")
    if not answers_file:
        answers_file = os.path.join(state.data_path, "answers.csv")
        logger.info(f"Using default answers file: {answers_file}")
    if not output_file:
        output_file = os.path.join(state.data_path, "agents_ranking.csv")
        logger.info(f"Using default output file: {output_file}")

    evaluations_file = os.path.join(state.data_path, "answers_eval.jsonl")
    reasonings_file = os.path.join(state.data_path, "reasonings.csv")

    doc_evaluator = DocumentEvaluatorFactory.create(
        document_evaluator_name,
        query_path=queries_file,
        documents_path=documents_file,
        output_file=reasonings_file,
        model_name=state.model_name,
        credentials_file=state.credentials_file,
        verbose=state.verbose,
        force=state.force,
    )

    doc_evaluator.annotate_all_docs()
    answer_evaluator = AnswerEvaluatorFactory.create(
        answer_evaluator_name,
        query_path=queries_file,
        answers_file=answers_file,
        reasonings_file=reasonings_file,
        output_file=evaluations_file,
        k=k,
        bidirectional=bidirectional,
        model_name=state.model_name,
        credentials_file=state.credentials_file,
        print_answers=state.verbose,
        force=state.force,
    )
    answer_evaluator.evaluate_all_answers()

    agent_ranker = AnswerRankerFactory.create(
        name=answer_ranker_name,
        output_file=output_file,
        evaluations_file=evaluations_file,
        initial_score=initial_score,
        k=elo_k,
        print=state.verbose,
        force=state.force,
    )
    agent_ranker.evaluate()
    agent_ranker.print_ranking()


def _version_callback(value: bool) -> None:
    if value:
        print(f"{__app_name__} v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    silent: Annotated[
        bool,
        typer.Option(
            "--silent",
            "-S",
            help="Run in silent mode, with minimal output",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite output files",
        ),
    ] = False,
    credentials_file: Annotated[
        Optional[str],
        typer.Option(
            "--credentials",
            "-c",
            help="Path to a file with OpenAI credentials. "
            "If missing, will use environment variables",
        ),
    ] = None,
    model_name: Annotated[
        str,
        typer.Option(
            "--model",
            "-m",
            help="Model to use as annotator",
        ),
    ] = "gpt-4",
    data_path: Annotated[
        str,
        typer.Option(
            "--data-path",
            help="Base path to store output data",
        ),
    ] = "data/",
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="Show the CLI version",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
):
    """A CLI for auto-eval"""
    state.verbose = not silent
    state.force = force
    state.credentials_file = credentials_file
    state.model_name = model_name
    state.data_path = data_path.rstrip("/")
