import typer

from ragelo.agent_rankers import AgentRankerFactory
from ragelo.cli.args import get_params_from_function
from ragelo.cli.utils import get_path
from ragelo.types import EloAgentRankerConfig

typer.main.get_params_from_function = get_params_from_function
app = typer.Typer()


@app.command()
def EloRanker(config: EloAgentRankerConfig = EloAgentRankerConfig(), **kwargs):
    """Evaluate the performance of multiple agents using an Elo ranking system."""

    config = EloAgentRankerConfig(**kwargs)
    config.output_file = get_path(config.data_path, config.output_file)
    config.evaluations_file = get_path(config.data_path, config.evaluations_file)
    ranker = AgentRankerFactory.create("elo", config)
    ranker.run()
