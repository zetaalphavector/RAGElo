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
from ragelo.utils import load_retrieved_docs_from_csv

typer.main.get_params_from_function = get_params_from_function


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
    config.query_path = get_path(config.data_path, config.query_path)
    config.documents_path = get_path(config.data_path, config.documents_path)
    config.answers_evaluations_path = get_path(
        config.data_path, config.answers_evaluations_path
    )

    config.verbose = True

    llm_provider = get_llm_provider(config.llm_provider, **kwargs)
    evaluator = get_retrieval_evaluator(
        RetrievalEvaluatorTypes.DOMAIN_EXPERT, config=config, llm_provider=llm_provider
    )
    documents = load_retrieved_docs_from_csv(
        config.documents_path, queries=config.query_path
    )
    evaluator.batch_evaluate(documents)


@app.command()
def reasoner(config: ReasonerEvaluatorConfig = ReasonerEvaluatorConfig(), **kwargs):
    """
    A document Evaluator that only outputs the reasoning for why a document is relevant.
    """
    config = ReasonerEvaluatorConfig(**kwargs)
    config.query_path = get_path(config.data_path, config.query_path)
    config.documents_path = get_path(config.data_path, config.documents_path)
    config.answers_evaluations_path = get_path(
        config.data_path, config.answers_evaluations_path
    )

    config.verbose = True
    llm_provider = get_llm_provider(config.llm_provider, **kwargs)
    evaluator = get_retrieval_evaluator(
        RetrievalEvaluatorTypes.REASONER, config=config, llm_provider=llm_provider
    )
    documents = load_retrieved_docs_from_csv(
        config.documents_path, queries=config.query_path
    )
    evaluator.batch_evaluate(documents)


@app.command()
def rdnam(config: RDNAMEvaluatorConfig = RDNAMEvaluatorConfig(), **kwargs):
    """Evaluator based on the paper by Thomas, Spielman, Craswell and Mitra, Large language models can accurately predict searcher preferences."""
    config = RDNAMEvaluatorConfig(**kwargs)
    config.query_path = get_path(config.data_path, config.query_path)
    config.documents_path = get_path(config.data_path, config.documents_path)
    config.answers_evaluations_path = get_path(
        config.data_path, config.answers_evaluations_path
    )

    config.verbose = True
    llm_provider = get_llm_provider(config.llm_provider, **kwargs)
    evaluator = get_retrieval_evaluator(
        RetrievalEvaluatorTypes.RDNAM, config=config, llm_provider=llm_provider
    )
    documents = load_retrieved_docs_from_csv(
        config.documents_path, queries=config.query_path
    )
    evaluator.batch_evaluate(documents)


if __name__ == "__main__":
    app()
