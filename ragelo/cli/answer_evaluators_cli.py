from __future__ import annotations

import logging

import typer

from ragelo import (
    Experiment,
    get_answer_evaluator,
    get_llm_provider,
    get_retrieval_evaluator,
)
from ragelo.cli.args import get_params_from_function
from ragelo.cli.utils import get_path
from ragelo.types import AnswerEvaluatorTypes
from ragelo.types.configurations.cli_configs import (
    CLIPairwiseDomainExpertEvaluatorConfig,
    CLIPairwiseEvaluatorConfig,
)
from ragelo.types.types import RetrievalEvaluatorTypes

typer.main.get_params_from_function = get_params_from_function  # type: ignore
app = typer.Typer()


@app.command()
def pairwise(config: CLIPairwiseEvaluatorConfig = CLIPairwiseEvaluatorConfig(), **kwargs):
    """An evaluator that evaluates RAG-based answers by comparing the answers of two agents to the same queries.

    example:
    >> ragelo answer-evaluator pairwise queries.csv answers.csv

    """
    logging.getLogger("ragelo").setLevel(logging.INFO)
    kwargs.pop("llm_answer_format", None)
    kwargs.pop("llm_response_schema", None)

    config = CLIPairwiseEvaluatorConfig(**kwargs)
    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)

    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    documents_file = get_path(config.data_dir, config.documents_csv_file)
    answers_file = get_path(config.data_dir, config.answers_csv_file)
    output_file = get_path(config.data_dir, config.output_file, check_exists=False) if config.output_file else None

    experiment = Experiment(
        experiment_name=config.experiment_name,
        save_path=output_file,
        queries_csv_path=queries_csv_file,
        documents_csv_path=documents_file,
        answers_csv_path=answers_file,
        verbose=config.verbose,
        clear_evaluations=config.force,
        rich_print=config.rich_print,
        cache_evaluations=config.save_results,
    )

    kwargs = config.model_dump()

    if config.add_reasoning:
        reasoner_evaluator = get_retrieval_evaluator(
            RetrievalEvaluatorTypes.REASONER,
            llm_provider=llm_provider,
            rich_print=config.rich_print,
            verbose=config.verbose,
        )
        reasoner_evaluator.evaluate_experiment(experiment)
        config.include_annotations = True
        config.include_raw_documents = False
    else:
        config.include_annotations = False
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
    logging.getLogger("ragelo").setLevel(logging.INFO)
    kwargs.pop("llm_answer_format", None)
    kwargs.pop("llm_response_schema", None)

    config = CLIPairwiseDomainExpertEvaluatorConfig(**kwargs)

    llm_provider = get_llm_provider(config.llm_provider_name, **kwargs)

    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    documents_file = get_path(config.data_dir, config.documents_csv_file)
    answers_file = get_path(config.data_dir, config.answers_csv_file)
    output_file = get_path(config.data_dir, config.output_file, check_exists=False) if config.output_file else None

    experiment = Experiment(
        experiment_name=config.experiment_name,
        save_path=output_file,
        queries_csv_path=queries_csv_file,
        documents_csv_path=documents_file,
        answers_csv_path=answers_file,
        verbose=config.verbose,
        clear_evaluations=config.force,
        rich_print=config.rich_print,
        cache_evaluations=config.save_results,
    )

    if config.add_reasoning:
        reasoner_evaluator = get_retrieval_evaluator(RetrievalEvaluatorTypes.REASONER, llm_provider=llm_provider)
        reasoner_evaluator.evaluate_experiment(experiment)
        config.include_annotations = True

    evaluator = get_answer_evaluator(AnswerEvaluatorTypes.PAIRWISE, config=config, llm_provider=llm_provider)
    evaluator.evaluate_experiment(experiment)
    experiment.save(output_file)
