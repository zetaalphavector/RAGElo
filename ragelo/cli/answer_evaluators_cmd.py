import os

import typer

from ragelo.cli.args import get_params_from_function
from ragelo.evaluators.answer_evaluators import AnswerEvaluatorFactory
from ragelo.llm_providers import LLMProviderFactory
from ragelo.types.configurations import AnswerEvaluatorConfig
from ragelo.utils import load_answers_from_csv, load_queries_from_csv

typer.main.get_params_from_function = get_params_from_function  # type: ignore
app = typer.Typer()


@app.command()
def pairwise_reasoning(
    config: AnswerEvaluatorConfig = AnswerEvaluatorConfig(), **kwargs
):
    """A evaluator that evaluates RAG-based answers pairwise, with document reasoning"""
    if kwargs["output_file"] is None:
        kwargs["output_file"] = os.path.join(
            kwargs.get("data_dir", ""), "answers_eval.jsonl"
        )
    config = AnswerEvaluatorConfig(**kwargs)
    llm_provider = LLMProviderFactory.create_from_credentials_file(
        config.llm_provider, config.credentials_file, config.model_name
    )
    evaluator = AnswerEvaluatorFactory.create(
        "pairwise_reasoning", llm_provider, config=config
    )
    queries = load_queries_from_csv(config.query_path)
    answers = load_answers_from_csv(config.answers_file, queries=queries)
    evaluator.run(answers)
