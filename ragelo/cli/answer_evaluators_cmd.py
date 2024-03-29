import typer

from ragelo import get_answer_evaluator, get_llm_provider
from ragelo.cli.args import get_params_from_function
from ragelo.types.configurations import PairWiseEvaluatorConfig
from ragelo.utils import load_answers_from_csv

typer.main.get_params_from_function = get_params_from_function  # type: ignore
app = typer.Typer()


@app.command()
def pairwise_reasoning(
    config: PairWiseEvaluatorConfig = PairWiseEvaluatorConfig(), **kwargs
):
    """A evaluator that evaluates RAG-based answers pairwise, with document reasoning"""
    config = PairWiseEvaluatorConfig(**kwargs)
    llm_provider = get_llm_provider(config.llm_provider, **kwargs)
    evaluator = get_answer_evaluator(
        "pairwise_reasoning", config=config, llm_provider=llm_provider
    )
    answers = load_answers_from_csv(config.answers_file, queries=config.query_path)
    evaluator.run(answers)
