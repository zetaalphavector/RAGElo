import typer

from ragelo import get_llm_provider, get_retrieval_evaluator
from ragelo.cli.args import get_params_from_function
from ragelo.types.configurations import (
    BaseEvaluatorConfig,
    DomainExpertEvaluatorConfig,
    RDNAMEvaluatorConfig,
)
from ragelo.utils import load_documents_from_csv

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
    llm_provider = get_llm_provider(config.llm_provider, **kwargs)
    evaluator = get_retrieval_evaluator(
        "domain_expert", config=config, llm_provider=llm_provider
    )
    documents = load_documents_from_csv(
        config.documents_path, queries=config.query_path
    )

    evaluator.run(documents)


@app.command()
def reasoner(config: BaseEvaluatorConfig = BaseEvaluatorConfig(), **kwargs):
    """
    A document Evaluator that only outputs the reasoning for why a document is relevant.
    """
    config = BaseEvaluatorConfig(**kwargs)
    llm_provider = get_llm_provider(config.llm_provider, **kwargs)
    evaluator = get_retrieval_evaluator(
        "reasoner", config=config, llm_provider=llm_provider
    )
    documents = load_documents_from_csv(
        config.documents_path, queries=config.query_path
    )

    evaluator.run(documents)


@app.command()
def rdnam(config: RDNAMEvaluatorConfig = RDNAMEvaluatorConfig(), **kwargs):
    """Evaluator based on the paper by Thomas, Spielman, Craswell and Mitra, Large language models can accurately predict searcher preferences."""
    config = RDNAMEvaluatorConfig(**kwargs)
    llm_provider = get_llm_provider(config.llm_provider, **kwargs)
    evaluator = get_retrieval_evaluator(
        "rdnam", config=config, llm_provider=llm_provider
    )
    documents = load_documents_from_csv(
        config.documents_path, queries=config.query_path
    )
    evaluator.run(documents)


if __name__ == "__main__":
    app()
