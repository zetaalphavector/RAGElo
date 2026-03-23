import typer

from ragelo import Experiment, get_agent_ranker, get_answer_evaluator, get_llm_provider, get_retrieval_evaluator
from ragelo.cli.answer_evaluators_cli import app as answer_evaluator_app
from ragelo.cli.args import get_params_from_function
from ragelo.cli.retrieval_evaluator_cli import app as retrieval_evaluator_app
from ragelo.cli.utils import build_storage_backend, get_path
from ragelo.logger import configure_logging
from ragelo.types import CLIConfig

typer.main.get_params_from_function = get_params_from_function  # type: ignore


app = typer.Typer()


app.add_typer(retrieval_evaluator_app, name="retrieval-evaluator")
app.add_typer(answer_evaluator_app, name="answer-evaluator")


@app.command()
def run_all(config: CLIConfig = CLIConfig(), **kwargs):
    """Run all the commands."""
    config = CLIConfig(**kwargs)
    configure_logging(level="INFO", rich=config.rich_print)

    # Parse the LLM provider and remove it from the kwargs
    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)

    # Get the absolute paths for the input and output files, and ensure that they exist.
    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    documents_file = get_path(config.data_dir, config.documents_csv_file)
    answers_file = get_path(config.data_dir, config.answers_csv_file)
    output_file = get_path(config.data_dir, config.output_file, check_exists=False) if config.output_file else None

    experiment = Experiment(
        experiment_name=config.experiment_name,
        storage_backend=build_storage_backend(config.experiment_name, output_file),
        queries_csv_path=queries_csv_file,
        documents_csv_path=documents_file,
        answers_csv_path=answers_file,
        show_results=config.show_results,
        clear_evaluations=config.force,
        rich_print=config.rich_print,
    )

    kwargs = config.model_dump()
    kwargs.pop("llm_response_schema", None)

    retrieval_evaluator = get_retrieval_evaluator("reasoner", llm_provider=llm_provider, **kwargs)
    answers_evaluator = get_answer_evaluator("pairwise", llm_provider=llm_provider, **kwargs)
    ranker = get_agent_ranker("elo", **kwargs)

    retrieval_evaluator.evaluate_experiment(experiment)
    answers_evaluator.evaluate_experiment(experiment)
    ranker.run(experiment=experiment)
    experiment.save(output_file)


if __name__ == "__main__":
    app()
