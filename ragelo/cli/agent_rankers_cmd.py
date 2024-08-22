import typer

from ragelo import get_agent_ranker
from ragelo.cli.args import get_params_from_function
from ragelo.cli.utils import get_path
from ragelo.types import EloAgentRankerConfig

typer.main.get_params_from_function = get_params_from_function
app = typer.Typer()


@app.command()
def EloRanker(config: EloAgentRankerConfig = EloAgentRankerConfig(), **kwargs):
    """Evaluate the performance of multiple agents using an Elo ranking system."""
    config.evaluations_file = get_path(config.data_dir, config.evaluations_file)
    config.agents_evaluations_file = get_path(
        config.data_dir, config.agents_evaluations_file
    )
    ranker = get_agent_ranker(config.ranker_name, **kwargs)

    ranker.run()
