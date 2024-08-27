import typer

from ragelo import get_answer_evaluator, get_llm_provider
from ragelo.cli.args import get_params_from_function
from ragelo.cli.utils import get_path
from ragelo.types import AnswerEvaluatorTypes
from ragelo.types.configurations import PairwiseEvaluatorConfig
from ragelo.utils import (
    add_answers_from_csv,
    add_documents_from_csv,
    load_queries_from_csv,
)

typer.main.get_params_from_function = get_params_from_function  # type: ignore
app = typer.Typer()


@app.command()
def pairwise_reasoning(
    config: PairwiseEvaluatorConfig = PairwiseEvaluatorConfig(), **kwargs
):
    """A evaluator that evaluates RAG-based answers pairwise, with document reasoning"""
    config = PairwiseEvaluatorConfig(**kwargs)
    config.queries_file = get_path(config.data_dir, config.queries_file)
    config.answers_file = get_path(config.data_dir, config.answers_file)
    config.documents_file = get_path(config.data_dir, config.documents_file)
    config.document_evaluations_file = get_path(
        config.data_dir, config.document_evaluations_file
    )

    config.games_evaluations_file = get_path(
        config.data_dir,
        config.games_evaluations_file,
        check_exists=False,
    )

    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)
    evaluator = get_answer_evaluator(
        AnswerEvaluatorTypes.PAIRWISE,
        config=config,
        llm_provider=llm_provider,
    )
    queries = load_queries_from_csv(config.queries_file)
    queries = add_documents_from_csv(config.documents_file, queries=queries)
    queries = add_answers_from_csv(config.answers_file, queries=queries)
    evaluator.batch_evaluate(queries)
