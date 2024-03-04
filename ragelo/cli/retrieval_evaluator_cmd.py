import typer

from ragelo.cli.args import get_params_from_function
from ragelo.evaluators.retrieval_evaluators import RetrievalEvaluatorFactory
from ragelo.llm_providers import LLMProviderFactory
from ragelo.types.configurations import (
    BaseEvaluatorConfig,
    DomainExpertEvaluatorConfig,
    RDNAMEvaluatorConfig,
)

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
    if kwargs["output_file"] is None:
        kwargs["output_file"] = "domain_expert_evaluations.csv"
    config = DomainExpertEvaluatorConfig(**kwargs)
    llm_provider = LLMProviderFactory.create_from_credentials_file(
        config.llm_provider, config.credentials_file, config.model_name
    )
    evaluator = RetrievalEvaluatorFactory.create("domain_expert", config, llm_provider)
    evaluator.run()


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
    evaluator = RetrievalEvaluatorFactory.create("reasoner", config, llm_provider)
    evaluator.run()


@app.command()
def rdnam(config: RDNAMEvaluatorConfig = RDNAMEvaluatorConfig(), **kwargs):
    """Evaluator based on the paper by Thomas, Spielman, Craswell and Mitra, Large language models can accurately predict searcher preferences."""
    if kwargs["output_file"] is None:
        kwargs["output_file"] = "rdnam_evaluations.csv"
    config = RDNAMEvaluatorConfig(**kwargs)
    llm_provider = LLMProviderFactory.create_from_credentials_file(
        config.llm_provider, config.credentials_file, config.model_name
    )
    evaluator = RetrievalEvaluatorFactory.create("rdnam", config, llm_provider)
    evaluator.run()


if __name__ == "__main__":
    app()
