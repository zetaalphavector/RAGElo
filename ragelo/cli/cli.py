import os

import typer

from ragelo.agent_rankers import AgentRankerFactory
from ragelo.cli.agent_rankers_cmd import app as ranker_app
from ragelo.cli.answer_evaluators_cmd import app as answer_evaluator_app
from ragelo.cli.args import get_params_from_function
from ragelo.cli.retrieval_evaluator_cmd import app as retrieval_evaluator_app
from ragelo.evaluators.answer_evaluators import AnswerEvaluatorFactory
from ragelo.evaluators.retrieval_evaluators import RetrievalEvaluatorFactory
from ragelo.llm_providers import LLMProviderFactory
from ragelo.types import (
    AllConfig,
    AnswerEvaluatorConfig,
    BaseEvaluatorConfig,
    EloAgentRankerConfig,
)

typer.main.get_params_from_function = get_params_from_function

app = typer.Typer()


app.add_typer(retrieval_evaluator_app, name="retrieval-evaluator")
app.add_typer(answer_evaluator_app, name="answer-evaluator")
app.add_typer(ranker_app, name="agents-ranker")


@app.command()
def run_all(config: AllConfig = AllConfig(), **kwargs):
    """Run all the commands."""
    print("Running all commands")
    if kwargs["output_file"] is None:
        kwargs["output_file"] = os.path.join(
            kwargs.get("data_path", ""), "agents_ranking.csv"
        )
    config = AllConfig(**kwargs)
    if config.retrieval_evaluator_name != "reasoner":
        raise ValueError("The retrieval evaluator must be a reasoner")
    if config.answer_evaluator_name != "pairwise_reasoning":
        raise ValueError("The answer evaluator must be a pairwise_reasoning")
    if config.answer_ranker_name != "elo":
        raise ValueError("The answer ranker must be an elo")

    llm_provider = LLMProviderFactory.create_from_credentials_file(
        config.llm_provider, config.credentials_file, config.model_name
    )

    retrieval_evaluator_config = BaseEvaluatorConfig(
        force=config.force,
        rich_print=config.rich_print,
        verbose=config.verbose,
        output_file=config.reasoning_file,
        query_path=config.query_path,
        documents_path=config.documents_path,
    )
    retrieval_evaluator = RetrievalEvaluatorFactory.create(
        "reasoner", retrieval_evaluator_config, llm_provider
    )
    retrieval_evaluator.run()

    answer_evaluator_config = AnswerEvaluatorConfig(
        force=config.force,
        rich_print=config.rich_print,
        verbose=config.verbose,
        answers_file=config.answers_file,
        output_file=config.evaluations_file,
        query_path=config.query_path,
        reasoning_file=config.reasoning_file,
        documents_path=config.documents_path,
        bidirectional=config.bidirectional,
        k=config.k,
    )
    answer_evaluator = AnswerEvaluatorFactory.create(
        "pairwise_reasoning", answer_evaluator_config, llm_provider
    )
    answer_evaluator.run()

    ranker_config = EloAgentRankerConfig(
        force=config.force,
        rich_print=config.rich_print,
        verbose=config.verbose,
        output_file=config.output_file,
        evaluations_file=config.evaluations_file,
        k=config.elo_k,
        initial_score=config.initial_score,
    )
    ranker = AgentRankerFactory.create("elo", ranker_config)
    ranker.run()


if __name__ == "__main__":
    app()
