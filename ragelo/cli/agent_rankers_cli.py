import typer

from ragelo import Experiment, get_agent_ranker
from ragelo.cli.args import get_params_from_function
from ragelo.cli.utils import get_path
from ragelo.types import AgentRankerTypes
from ragelo.types.configurations.cli_configs import CLIEloAgentRankerConfig

typer.main.get_params_from_function = get_params_from_function  # type: ignore
app = typer.Typer()


@app.command()
def elo(config: CLIEloAgentRankerConfig = CLIEloAgentRankerConfig(), **kwargs):
    """Evaluate the performance of multiple agents using an Elo ranking system."""
    config = CLIEloAgentRankerConfig(**kwargs)
    queries_csv_file = get_path(config.data_dir, config.queries_csv_file)
    experiment = Experiment(experiment_name=config.experiment_name, queries_csv_path=queries_csv_file)
    ranker = get_agent_ranker(AgentRankerTypes.ELO, config=config)
    ranker.run(experiment=experiment)
