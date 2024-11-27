from __future__ import annotations

import typer

from ragelo import (
    Experiment,
    get_agent_ranker,
    get_answer_evaluator,
    get_llm_provider,
    get_retrieval_evaluator,
)
from ragelo.cli.agent_rankers_cmd import app as ranker_app
from ragelo.cli.answer_evaluators_cmd import app as answer_evaluator_app
from ragelo.cli.args import get_params_from_function
from ragelo.cli.retrieval_evaluator_cmd import app as retrieval_evaluator_app
from ragelo.cli.utils import get_path
from ragelo.types import CLIConfig

typer.main.get_params_from_function = get_params_from_function  # type: ignore


app = typer.Typer()


app.add_typer(retrieval_evaluator_app, name="retrieval-evaluator")
app.add_typer(answer_evaluator_app, name="answer-evaluator")
app.add_typer(ranker_app, name="agents-ranker")


@app.command()
def run_all(config: CLIConfig = CLIConfig(), **kwargs):
    """Run all the commands."""

    config = CLIConfig(**kwargs)

    # Parse the LLM provider and remove it from the kwargs
    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)

    # Get the absolute paths for the input and output files, and ensure that they exist.
    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    documents_file = get_path(config.data_dir, config.documents_csv_file)
    answers_file = get_path(config.data_dir, config.answers_csv_file)

    if config.force:
        clear_evaluations = True
    else:
        clear_evaluations = False

    experiment = Experiment(
        experiment_name=config.experiment_name,
        queries_csv_path=queries_csv_file,
        documents_csv_path=documents_file,
        answers_csv_path=answers_file,
        verbose=config.verbose,
        clear_evaluations=clear_evaluations,
        rich_print=config.rich_print,
        persist_on_disk=config.save_results,
    )

    kwargs = config.model_dump()
    kwargs.pop("llm_answer_format")
    kwargs.pop("llm_response_schema")

    retrieval_evaluator = get_retrieval_evaluator("reasoner", llm_provider=llm_provider, **kwargs)
    answers_evaluator = get_answer_evaluator("pairwise", llm_provider=llm_provider, **kwargs)
    ranker = get_agent_ranker("elo", **kwargs)

    retrieval_evaluator.evaluate_experiment(experiment)
    answers_evaluator.evaluate_experiment(experiment)
    ranker.run(experiment=experiment)
    experiment.save(output_path=config.output_file)


if __name__ == "__main__":
    app()
