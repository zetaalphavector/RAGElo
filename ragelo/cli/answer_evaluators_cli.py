import typer

from ragelo import Experiment, get_answer_evaluator, get_llm_provider, get_retrieval_evaluator
from ragelo.cli.args import get_params_from_function
from ragelo.cli.utils import get_path
from ragelo.logger import configure_logging
from ragelo.types import AnswerEvaluatorTypes
from ragelo.types.configurations.cli_configs import CLIPairwiseDomainExpertEvaluatorConfig, CLIPairwiseEvaluatorConfig
from ragelo.types.storage import build_storage_backend
from ragelo.types.types import RetrievalEvaluatorTypes

typer.main.get_params_from_function = get_params_from_function  # type: ignore
app = typer.Typer()


@app.command()
def pairwise(config: CLIPairwiseEvaluatorConfig = CLIPairwiseEvaluatorConfig(), **kwargs):
    """An evaluator that evaluates RAG-based answers by comparing the answers of two agents to the same queries.

    example:
    >> ragelo answer-evaluator pairwise queries.csv answers.csv

    """
    kwargs.pop("llm_response_schema", None)

    config = CLIPairwiseEvaluatorConfig(**kwargs)
    configure_logging(level="INFO", rich=config.rich_print)
    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)

    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    documents_file = get_path(config.data_dir, config.documents_csv_file)
    answers_file = get_path(config.data_dir, config.answers_csv_file)
    output_file = get_path(config.data_dir, config.output_file, check_exists=False) if config.output_file else None
    storage_backend = build_storage_backend(config.experiment_name, output_file)

    experiment = Experiment(
        experiment_name=config.experiment_name,
        storage_backend=storage_backend,
        queries_csv_path=queries_csv_file,
        documents_csv_path=documents_file,
        answers_csv_path=answers_file,
        show_results=config.show_results,
        clear_evaluations=config.force,
        rich_print=config.rich_print,
    )

    kwargs = config.model_dump()

    if config.add_reasoning:
        reasoner_evaluator = get_retrieval_evaluator(
            RetrievalEvaluatorTypes.REASONER,
            llm_provider=llm_provider,
            rich_print=config.rich_print,
            show_results=config.show_results,
        )
        reasoner_evaluator.evaluate_experiment(experiment)
        config.include_relevance_score = True
        config.include_raw_documents = False
    else:
        config.include_relevance_score = False
        config.include_raw_documents = True

    evaluator = get_answer_evaluator(AnswerEvaluatorTypes.PAIRWISE, config=config, llm_provider=llm_provider)
    evaluator.evaluate_experiment(experiment)
    experiment.save(output_file)


@app.command()
def expert_pairwise(
    config: CLIPairwiseDomainExpertEvaluatorConfig = CLIPairwiseDomainExpertEvaluatorConfig(),
    **kwargs,
):
    """
    An evaluator that evaluates RAG-based answers by comparing answers of two agents and impersonating a domain expert.
    """
    kwargs.pop("llm_response_schema", None)

    config = CLIPairwiseDomainExpertEvaluatorConfig(**kwargs)
    configure_logging(level="INFO", rich=config.rich_print)

    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)

    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    documents_file = get_path(config.data_dir, config.documents_csv_file)
    answers_file = get_path(config.data_dir, config.answers_csv_file)
    output_file = get_path(config.data_dir, config.output_file, check_exists=False) if config.output_file else None
    storage_backend = build_storage_backend(config.experiment_name, output_file)

    experiment = Experiment(
        experiment_name=config.experiment_name,
        storage_backend=storage_backend,
        queries_csv_path=queries_csv_file,
        documents_csv_path=documents_file,
        answers_csv_path=answers_file,
        show_results=config.show_results,
        clear_evaluations=config.force,
        rich_print=config.rich_print,
    )

    if config.add_reasoning:
        reasoner_evaluator = get_retrieval_evaluator(
            RetrievalEvaluatorTypes.REASONER,
            llm_provider=llm_provider,
            rich_print=config.rich_print,
            show_results=config.show_results,
            use_progress_bar=config.use_progress_bar,
        )
        reasoner_evaluator.evaluate_experiment(experiment)
        config.include_relevance_score = True

    evaluator = get_answer_evaluator(AnswerEvaluatorTypes.PAIRWISE, config=config, llm_provider=llm_provider)
    evaluator.evaluate_experiment(experiment)
    experiment.save(output_file)
