import typer
from rich.console import Console

from ragelo.benchmarks.datasets import get_dataset
from ragelo.benchmarks.pricing import Price
from ragelo.cli.args import config_command
from ragelo.cli.utils import config_kwargs, get_cli_llm_provider
from ragelo.logger import configure_logging
from ragelo.types import BaseConfig
from ragelo.types.configurations.cli_configs import CLIBenchmarkConfig, CLITrecRag24AnswersBenchmarkConfig
from ragelo.types.types import BenchmarkDatasetTypes

app = typer.Typer()


def run_benchmark(dataset_name: BenchmarkDatasetTypes, config: CLIBenchmarkConfig) -> None:
    """Judges the dataset with every evaluator on every model, one cached experiment each, and prints how the
    judgments compare with the human ones."""
    configure_logging(level="WARNING", rich=config.rich_print)
    if not config.models or not config.evaluators:
        raise typer.BadParameter("Give at least one --models and one --evaluators.")
    try:
        prices = dict(Price.parse(spec) for spec in config.prices or [])
        dataset = get_dataset(dataset_name, config=config)
    except ValueError as error:
        raise typer.BadParameter(str(error))
    dataset.download()
    output_dir = config.output_dir or f"benchmarks/results/{dataset_name}"

    outcomes = {}
    for model in config.models:
        llm_provider = get_cli_llm_provider(config.llm_provider_name, {"model": model})
        for variant in config.evaluators:
            evaluator = dataset.get_evaluator(variant, llm_provider, **config_kwargs(BaseConfig, config.model_dump()))
            experiment_name = f"{dataset.subset}_{variant}_{model.replace('/', '_')}{config.tag and '_' + config.tag}"
            outcomes[(variant, model)] = dataset.judge(evaluator, experiment_name, output_dir)

    console = Console()
    for table in dataset.tables(outcomes, prices):
        console.print(table)


@app.command()
@config_command
def llmjudge(config: CLIBenchmarkConfig):
    """Retrieval evaluators against the NIST 0-3 labels of LLMJudge: TREC DL 2023 passages, dev and test splits."""
    run_benchmark(BenchmarkDatasetTypes.LLMJUDGE, config)


@app.command()
@config_command
def llmjudge_pairwise(config: CLIBenchmarkConfig):
    """Pairwise answer evaluators on games between two LLMJudge passages of one query: how often the passage
    the assessors graded higher wins. Judges 500 games unless --n-samples says otherwise."""
    run_benchmark(BenchmarkDatasetTypes.LLMJUDGE_PAIRWISE, config)


@app.command()
@config_command
def trec_rag24(config: CLIBenchmarkConfig):
    """Retrieval evaluators against the NIST 0-3 labels of the TREC 2024 RAG retrieval task.

    The first run streams the 27 GB MS MARCO v2.1 segment corpus and stores only the judged segments, 28 MB.
    """
    run_benchmark(BenchmarkDatasetTypes.TREC_RAG24, config)


@app.command()
@config_command
def trec_rag24_answers(config: CLITrecRag24AnswersBenchmarkConfig):
    """An Elo tournament of pairwise answer evaluators between systems of the TREC 2024 RAG track, against
    their ranking by NIST's nugget scores. Evaluators with "rubric" in their name get each topic's nuggets as
    its rubric. --n-samples draws a difficulty-stratified sample of topics."""
    run_benchmark(BenchmarkDatasetTypes.TREC_RAG24_ANSWERS, config)
