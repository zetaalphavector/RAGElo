import typer

from ragelo import Experiment, get_llm_provider, get_retrieval_evaluator
from ragelo.cli.args import get_params_from_function
from ragelo.cli.utils import get_path
from ragelo.types.configurations.cli_configs import (
    CLIDomainExpertEvaluatorConfig,
    CLIRDNAMEvaluatorConfig,
    CLIReasonerEvaluatorConfig,
)
from ragelo.types.types import RetrievalEvaluatorTypes

typer.main.get_params_from_function = get_params_from_function  # type: ignore


app = typer.Typer()


@app.command()
def domain_expert(config: CLIDomainExpertEvaluatorConfig = CLIDomainExpertEvaluatorConfig(), **kwargs):
    """Evaluator with a domain expert persona.

    This Retrieval Evaluator evaluates the relevance of documents submitted by
    a user that is an expert in a specific domain. For instance,
    to evaluate the documents retrieved to queries submitted by a Chemical
    Engineer that works at ChemCorp Inc:

    ragelo retrieval_evaluator domain_expert queries.csv documents.csv "Chemical Engineering" --company "ChemCorp Inc."

    """

    config = CLIDomainExpertEvaluatorConfig(**kwargs)
    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    documents_file = get_path(config.data_dir, config.documents_csv_file)

    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)
    experiment = Experiment(experiment_name=config.experiment_name, queries_csv_path=queries_csv_file)
    experiment.add_retrieved_docs_from_csv(documents_file)
    evaluator = get_retrieval_evaluator(
        RetrievalEvaluatorTypes.DOMAIN_EXPERT, config=config, llm_provider=llm_provider
    )
    evaluator.evaluate_experiment(experiment)


@app.command()
def reasoner(config: CLIReasonerEvaluatorConfig = CLIReasonerEvaluatorConfig(), **kwargs):
    """
    A document Evaluator that only outputs the reasoning for why a document is relevant.
    """
    config = CLIReasonerEvaluatorConfig(**kwargs)
    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    documents_file = get_path(config.data_dir, config.documents_csv_file)

    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)
    experiment = Experiment(experiment_name=config.experiment_name, queries_csv_path=queries_csv_file)
    experiment.add_retrieved_docs_from_csv(documents_file)
    evaluator = get_retrieval_evaluator(RetrievalEvaluatorTypes.REASONER, config=config, llm_provider=llm_provider)
    evaluator.evaluate_experiment(experiment)


@app.command()
def rdnam(config: CLIRDNAMEvaluatorConfig = CLIRDNAMEvaluatorConfig(), **kwargs):
    """
    Evaluator based on the paper by Thomas, Spielman, Craswell and Mitra:
    Large language models can accurately predict searcher preferences.
    """
    config = CLIRDNAMEvaluatorConfig(**kwargs)
    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    documents_file = get_path(config.data_dir, config.documents_csv_file)

    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)
    experiment = Experiment(experiment_name=config.experiment_name, queries_csv_path=queries_csv_file)
    experiment.add_retrieved_docs_from_csv(documents_file)
    evaluator = get_retrieval_evaluator(RetrievalEvaluatorTypes.RDNAM, config=config, llm_provider=llm_provider)
    evaluator.evaluate_experiment(experiment)


if __name__ == "__main__":
    app()
