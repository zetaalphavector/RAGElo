import typer

from ragelo import get_answer_evaluator, get_llm_provider, get_retrieval_evaluator
from ragelo.agent_rankers import AgentRankerFactory
from ragelo.cli.agent_rankers_cmd import app as ranker_app
from ragelo.cli.answer_evaluators_cmd import app as answer_evaluator_app
from ragelo.cli.args import get_params_from_function
from ragelo.cli.retrieval_evaluator_cmd import app as retrieval_evaluator_app
from ragelo.cli.utils import get_path
from ragelo.types import AllConfig, EloAgentRankerConfig
from ragelo.utils import load_answers_from_csv, load_retrieved_docs_from_csv

typer.main.get_params_from_function = get_params_from_function

app = typer.Typer()


app.add_typer(retrieval_evaluator_app, name="retrieval-evaluator")
app.add_typer(answer_evaluator_app, name="answer-evaluator")
app.add_typer(ranker_app, name="agents-ranker")


@app.command()
def run_all(config: AllConfig = AllConfig(), **kwargs):
    """Run all the commands."""

    config = AllConfig(**kwargs)
    if config.retrieval_evaluator_name != "reasoner":
        raise ValueError("The retrieval evaluator must be a reasoner")
    if config.answer_evaluator_name != "pairwise_reasoning":
        raise ValueError("The answer evaluator must be a pairwise_reasoning")
    if config.answer_ranker_name != "elo":
        raise ValueError("The answer ranker must be an elo")
    config.verbose = True

    llm_provider = get_llm_provider(config.llm_provider, **kwargs)

    args_to_remove = {
        "output_file",
        "llm_provider",
    }

    # Clean the data paths
    config.query_path = get_path(config.data_path, config.query_path)
    config.documents_path = get_path(config.data_path, config.documents_path)
    config.answers_path = get_path(config.data_path, config.answers_path)
    config.reasoning_path = get_path(config.data_path, config.reasoning_path)
    config.evaluations_file = get_path(config.data_path, config.evaluations_file)

    args_clean = {
        k: v for k, v in config.model_dump().items() if k not in args_to_remove
    }

    retrieval_evaluator = get_retrieval_evaluator(
        "reasoner",
        llm_provider=llm_provider,
        output_file=config.reasoning_path,
        **args_clean,
    )
    documents = load_retrieved_docs_from_csv(
        documents_path=config.documents_path, queries=config.query_path
    )
    answers = load_answers_from_csv(
        answers_path=config.answers_path, queries=config.query_path
    )
    retrieval_evaluator.batch_evaluate(documents)

    answers_evaluator = get_answer_evaluator(
        "pairwise_reasoning",
        llm_provider=llm_provider,
        output_file=config.evaluations_file,
        **args_clean,
    )
    answers_evaluator.batch_evaluate(answers)

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
