import typer

from ragelo import get_llm_provider, get_retrieval_evaluator
from ragelo.cli.args import get_params_from_function
from ragelo.cli.utils import get_path
from ragelo.types.configurations import (
    DomainExpertEvaluatorConfig,
    RDNAMEvaluatorConfig,
    ReasonerEvaluatorConfig,
)
from ragelo.types.types import RetrievalEvaluatorTypes
from ragelo.utils import add_documents_from_csv, load_queries_from_csv

typer.main.get_params_from_function = get_params_from_function  # type: ignore


app = typer.Typer()


@app.command()
def domain_expert(
    config: DomainExpertEvaluatorConfig = DomainExpertEvaluatorConfig(), **kwargs
):
    """Evaluator with a domain expert persona.

    This Retrieval Evaluator evaluates the relevance of documents submitted by
    a user that is an expert in a specific domain. For instance,
    to evaluate the documents retrieved to queries submitted by a Chemical
    Engineer that works at ChemCorp Inc:

    ragelo retrieval_evaluator domain_expert queries.csv documents.csv "Chemical Engineering" --company "ChemCorp Inc."

    """

    config = DomainExpertEvaluatorConfig(**kwargs)
    config.queries_file = get_path(config.data_dir, config.queries_file)
    config.documents_file = get_path(config.data_dir, config.documents_file)
    config.document_evaluations_file = get_path(
        config.data_dir, config.document_evaluations_file, check_exists=False
    )

    config.verbose = True

    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)
    evaluator = get_retrieval_evaluator(
        RetrievalEvaluatorTypes.DOMAIN_EXPERT, config=config, llm_provider=llm_provider
    )
    queries = load_queries_from_csv(config.queries_file)
    queries = add_documents_from_csv(config.documents_file, queries=queries)
    evaluator.batch_evaluate(queries)


@app.command()
def reasoner(config: ReasonerEvaluatorConfig = ReasonerEvaluatorConfig(), **kwargs):
    """
    A document Evaluator that only outputs the reasoning for why a document is relevant.
    """
    config = ReasonerEvaluatorConfig(**kwargs)
    config.queries_file = get_path(config.data_dir, config.queries_file)
    config.documents_file = get_path(config.data_dir, config.documents_file)
    config.document_evaluations_file = get_path(
        config.data_dir, config.document_evaluations_file, check_exists=False
    )

    config.verbose = True

    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)
    evaluator = get_retrieval_evaluator(
        RetrievalEvaluatorTypes.REASONER, config=config, llm_provider=llm_provider
    )
    queries = load_queries_from_csv(config.queries_file)
    queries = add_documents_from_csv(config.documents_file, queries=queries)
    evaluator.batch_evaluate(queries)


@app.command()
def rdnam(config: RDNAMEvaluatorConfig = RDNAMEvaluatorConfig(), **kwargs):
    """Evaluator based on the paper by Thomas, Spielman, Craswell and Mitra, Large language models can accurately predict searcher preferences."""
    config = RDNAMEvaluatorConfig(**kwargs)
    config.queries_file = get_path(config.data_dir, config.queries_file)
    config.documents_file = get_path(config.data_dir, config.documents_file)
    config.document_evaluations_file = get_path(
        config.data_dir, config.document_evaluations_file, check_exists=False
    )
    config.verbose = True
    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)
    evaluator = get_retrieval_evaluator(
        RetrievalEvaluatorTypes.RDNAM, config=config, llm_provider=llm_provider
    )
    queries = load_queries_from_csv(config.queries_file)
    queries = add_documents_from_csv(config.documents_file, queries=queries)
    evaluator.batch_evaluate(queries)


if __name__ == "__main__":
    app()
