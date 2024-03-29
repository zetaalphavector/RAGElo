import os

import typer

from ragelo import get_answer_evaluator, get_llm_provider, get_retrieval_evaluator
from ragelo.agent_rankers import AgentRankerFactory
from ragelo.cli.agent_rankers_cmd import app as ranker_app
from ragelo.cli.answer_evaluators_cmd import app as answer_evaluator_app
from ragelo.cli.args import get_params_from_function
from ragelo.cli.retrieval_evaluator_cmd import app as retrieval_evaluator_app
from ragelo.types import (
    AllConfig,
    BaseAnswerEvaluatorConfig,
    BaseEvaluatorConfig,
    EloAgentRankerConfig,
)
from ragelo.utils import (
    load_answers_from_csv,
    load_documents_from_csv,
    load_queries_from_csv,
)

typer.main.get_params_from_function = get_params_from_function  # type: ignore

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
    llm_provider = get_llm_provider(config.llm_provider, **kwargs)

    args_without_output = {k: v for k, v in kwargs.items() if k != "output_file"}
    retrieval_evaluator_config = BaseEvaluatorConfig(
        output_file=config.reasoning_file, **args_without_output
    )
    retrieval_evaluator = get_retrieval_evaluator(
        config.retrieval_evaluator_name,
        config=retrieval_evaluator_config,
        llm_provider=llm_provider,
    )

    queries = load_queries_from_csv(retrieval_evaluator_config.query_path)
    documents = load_documents_from_csv(
        retrieval_evaluator_config.documents_path, queries=queries
    )
    retrieval_evaluator.run(documents)
    answer_evaluator_config = BaseAnswerEvaluatorConfig(
        output_file=config.evaluations_file, **args_without_output
    )
    answer_evaluator = get_answer_evaluator(
        config.answer_evaluator_name,
        config=answer_evaluator_config,
        llm_provider=llm_provider,
    )

    answers = load_answers_from_csv(
        answer_evaluator_config.answers_file, queries=queries
    )
    answer_evaluator.run(answers)

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
