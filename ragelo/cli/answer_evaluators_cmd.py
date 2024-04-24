import asyncio

import typer

from ragelo import get_answer_evaluator, get_llm_provider
from ragelo.cli.args import get_params_from_function
from ragelo.cli.utils import get_path
from ragelo.types import AnswerEvaluatorTypes
from ragelo.types.configurations import PairwiseEvaluatorConfig
from ragelo.utils import load_answers_from_csv

typer.main.get_params_from_function = get_params_from_function
app = typer.Typer()


@app.command()
def pairwise_reasoning(
    config: PairwiseEvaluatorConfig = PairwiseEvaluatorConfig(), **kwargs
):
    """A evaluator that evaluates RAG-based answers pairwise, with document reasoning"""
    config = PairwiseEvaluatorConfig(**kwargs)
    config.query_path = get_path(config.data_path, config.query_path)
    config.answers_path = get_path(config.data_path, config.answers_path)
    config.output_file = get_path(config.data_path, config.output_file)
    config.documents_path = get_path(config.data_path, config.documents_path)

    llm_provider = get_llm_provider(config.llm_provider, **kwargs)
    evaluator = get_answer_evaluator(
        AnswerEvaluatorTypes.PAIRWISE_REASONING,
        config=config,
        llm_provider=llm_provider,
    )
    answers = load_answers_from_csv(config.answers_path, queries=config.query_path)
    with asyncio.Runner() as runner:
        runner.run(evaluator.batch_evaluate(answers))
