import typer

from ragelo.cli.args import get_params_from_function
from ragelo.evaluators.retrieval_evaluators import (
    BaseRetrievalEvaluator,
    RetrievalEvaluatorFactory,
)
from ragelo.llm_providers import BaseLLMProvider, LLMProviderFactory
from ragelo.types import Document
from ragelo.types.configurations import (
    BaseEvaluatorConfig,
    DomainExpertEvaluatorConfig,
    RDNAMEvaluatorConfig,
)
from ragelo.utils import load_documents_from_csv, load_queries_from_csv

typer.main.get_params_from_function = get_params_from_function  # type: ignore


app = typer.Typer()


def create_evaluator(
    evaluator_name: str,
    llm_provider: BaseLLMProvider,
    output_file: str,
    config: BaseEvaluatorConfig,
) -> BaseRetrievalEvaluator:
    return RetrievalEvaluatorFactory.create(
        evaluator_name, llm_provider, output_file=output_file, config=config
    )


def load_documents(
    query_path: str, documents_path: str
) -> dict[str, dict[str, Document]]:
    queries = load_queries_from_csv(query_path)
    documents = load_documents_from_csv(documents_path, queries=queries)
    return documents


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
    if kwargs["output_file"] is None:
        kwargs["output_file"] = "domain_expert_evaluations.csv"
    config = DomainExpertEvaluatorConfig(**kwargs)

    llm_provider = LLMProviderFactory.create_from_credentials_file(
        config.llm_provider, config.credentials_file, config.model_name
    )
    evaluator = create_evaluator(
        "domain_expert", llm_provider, output_file=config.output_file, **kwargs
    )
    documents = load_documents(config.query_path, config.documents_path)

    evaluator.run(documents)


@app.command()
def reasoner(config: BaseEvaluatorConfig = BaseEvaluatorConfig(), **kwargs):
    """
    A document Evaluator that only outputs the reasoning for why a document is relevant.
    """
    if kwargs["output_file"] is None:
        kwargs["output_file"] = "reasonings.csv"
    llm_provider = LLMProviderFactory.create_from_credentials_file(
        config.llm_provider, config.credentials_file, config.model_name
    )
    evaluator = create_evaluator(
        "reasoner", llm_provider, output_file=config.output_file, **kwargs
    )
    documents = load_documents(config.query_path, config.documents_path)

    evaluator.run(documents)


@app.command()
def rdnam(config: RDNAMEvaluatorConfig = RDNAMEvaluatorConfig(), **kwargs):
    """Evaluator based on the paper by Thomas, Spielman, Craswell and Mitra, Large language models can accurately predict searcher preferences."""
    if kwargs["output_file"] is None:
        kwargs["output_file"] = "rdnam_evaluations.csv"
    config = RDNAMEvaluatorConfig(**kwargs)
    llm_provider = LLMProviderFactory.create_from_credentials_file(
        config.llm_provider, config.credentials_file, config.model_name
    )
    evaluator = create_evaluator(
        "rdnam", llm_provider, output_file=config.output_file, **kwargs
    )
    documents = load_documents(config.query_path, config.documents_path)

    evaluator.run(documents)


if __name__ == "__main__":
    app()
