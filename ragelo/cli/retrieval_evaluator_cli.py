import typer

from ragelo import Experiment, get_retrieval_evaluator
from ragelo.cli.args import config_command
from ragelo.cli.utils import get_cli_llm_provider, get_path
from ragelo.logger import configure_logging
from ragelo.types.configurations.cli_configs import (
    CLIDomainExpertEvaluatorConfig,
    CLIRDNAMEvaluatorConfig,
    CLIReasonerEvaluatorConfig,
)
from ragelo.types.types import RetrievalEvaluatorTypes

app = typer.Typer()


@app.command()
@config_command
def domain_expert(config: CLIDomainExpertEvaluatorConfig):
    """Evaluator with a domain expert persona.

    This Retrieval Evaluator evaluates the relevance of documents submitted by
    a user that is an expert in a specific domain. For instance,
    to evaluate the documents retrieved to queries submitted by a Chemical
    Engineer that works at ChemCorp Inc:

    ragelo retrieval_evaluator domain_expert queries.csv documents.csv "Chemical Engineering" --company "ChemCorp Inc."

    """
    configure_logging(level="INFO", rich=config.rich_print)
    llm_provider = get_cli_llm_provider(config.llm_provider_name, config.model_dump())

    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    documents_file = get_path(config.data_dir, config.documents_csv_file)
    output_file = get_path(config.data_dir, config.output_file, check_exists=False) if config.output_file else None

    experiment = Experiment(
        experiment_name=config.experiment_name,
        save_path=output_file,
        queries_csv_path=queries_csv_file,
        documents_csv_path=documents_file,
        show_results=config.show_results,
        clear_evaluations=config.force,
        rich_print=config.rich_print,
    )

    evaluator = get_retrieval_evaluator(
        RetrievalEvaluatorTypes.DOMAIN_EXPERT, config=config, llm_provider=llm_provider
    )
    evaluator.evaluate_experiment(experiment)
    experiment.save(output_file)


@app.command()
@config_command
def reasoner(config: CLIReasonerEvaluatorConfig):
    """
    A document Evaluator that only outputs the reasoning for why a document is relevant.
    """
    configure_logging(level="INFO", rich=config.rich_print)

    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    documents_file = get_path(config.data_dir, config.documents_csv_file)
    output_file = get_path(config.data_dir, config.output_file, check_exists=False) if config.output_file else None

    experiment = Experiment(
        experiment_name=config.experiment_name,
        save_path=output_file,
        queries_csv_path=queries_csv_file,
        documents_csv_path=documents_file,
        show_results=config.show_results,
        clear_evaluations=config.force,
        rich_print=config.rich_print,
    )

    llm_provider = get_cli_llm_provider(config.llm_provider_name, config.model_dump())
    evaluator = get_retrieval_evaluator(RetrievalEvaluatorTypes.REASONER, config=config, llm_provider=llm_provider)
    evaluator.evaluate_experiment(experiment)
    experiment.save(output_file)


@app.command()
@config_command
def rdnam(config: CLIRDNAMEvaluatorConfig):
    """
    Evaluator based on the paper by Thomas, Spielman, Craswell and Mitra:
    Large language models can accurately predict searcher preferences.
    """
    configure_logging(level="INFO", rich=config.rich_print)
    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    documents_file = get_path(config.data_dir, config.documents_csv_file)
    output_file = get_path(config.data_dir, config.output_file, check_exists=False) if config.output_file else None

    experiment = Experiment(
        experiment_name=config.experiment_name,
        save_path=output_file,
        queries_csv_path=queries_csv_file,
        documents_csv_path=documents_file,
        show_results=config.show_results,
        clear_evaluations=config.force,
        rich_print=config.rich_print,
    )

    llm_provider = get_cli_llm_provider(config.llm_provider_name, config.model_dump())
    evaluator = get_retrieval_evaluator(RetrievalEvaluatorTypes.RDNAM, config=config, llm_provider=llm_provider)
    evaluator.evaluate_experiment(experiment)
    experiment.save(output_file)


if __name__ == "__main__":
    app()
