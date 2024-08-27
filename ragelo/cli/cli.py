import typer

from ragelo import (
    get_agent_ranker,
    get_answer_evaluator,
    get_llm_provider,
    get_retrieval_evaluator,
)
from ragelo.cli.agent_rankers_cmd import app as ranker_app
from ragelo.cli.answer_evaluators_cmd import app as answer_evaluator_app
from ragelo.cli.args import get_params_from_function
from ragelo.cli.retrieval_evaluator_cmd import app as retrieval_evaluator_app
from ragelo.cli.utils import get_path
from ragelo.types import AllConfig
from ragelo.utils import (
    add_answers_from_csv,
    add_documents_from_csv,
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

    config = AllConfig(**kwargs)
    if config.retrieval_evaluator_name != "reasoner":
        raise ValueError("The retrieval evaluator must be a reasoner")
    if config.answer_evaluator_name != "pairwise":
        raise ValueError("The answer evaluator must be pairwise")
    if config.answer_ranker_name != "elo":
        raise ValueError("The answer ranker must be an elo")
    config.verbose = True

    # Parse the LLM provider and remove it from the kwargs
    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)

    # Get the absolute paths for the input and output files, and ensure that they exist.
    # TODO: Make this more robust.
    config.queries_file = get_path(config.data_dir, config.queries_file)
    config.documents_file = get_path(config.data_dir, config.documents_file)
    config.answers_file = get_path(config.data_dir, config.answers_file)
    config.reasoning_file = get_path(
        config.data_dir, config.reasoning_file, check_exists=False
    )
    config.answers_evaluations_file = get_path(
        config.data_dir, config.answers_evaluations_file, check_exists=False
    )
    config.agents_evaluations_file = get_path(
        config.data_dir, config.agents_evaluations_file, check_exists=False
    )
    config.games_evaluations_file = get_path(
        config.data_dir, config.games_evaluations_file, check_exists=False
    )
    config.document_evaluations_file = get_path(
        config.data_dir, config.document_evaluations_file, check_exists=False
    )

    kwargs = config.model_dump()
    retrieval_evaluator = get_retrieval_evaluator(
        config.retrieval_evaluator_name, llm_provider=llm_provider, **kwargs
    )

    answers_evaluator = get_answer_evaluator(
        config.answer_evaluator_name, llm_provider=llm_provider, **kwargs
    )

    # TODO: Instead of managing a list of queries, we should have a Dataset type that can load the queries and documents directly.
    queries = load_queries_from_csv(config.queries_file)
    queries = add_documents_from_csv(
        documents_file=config.documents_file, queries=queries
    )
    queries = add_answers_from_csv(answers_file=config.answers_file, queries=queries)

    retrieval_evaluator.batch_evaluate(queries)
    answers_evaluator.batch_evaluate(queries)

    ranker = get_agent_ranker("elo", **kwargs)
    ranker.run(queries=queries)


if __name__ == "__main__":
    app()
