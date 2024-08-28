import typer

from ragelo import get_agent_ranker
from ragelo.cli.args import get_params_from_function
from ragelo.cli.utils import get_path
from ragelo.types import AgentRankerTypes, EloAgentRankerConfig
from ragelo.utils import load_answer_evaluations_from_csv

typer.main.get_params_from_function = get_params_from_function  # type: ignore
app = typer.Typer()


@app.command()
def elo(config: EloAgentRankerConfig = EloAgentRankerConfig(), **kwargs):
    """Evaluate the performance of multiple agents using an Elo ranking system."""
    config = EloAgentRankerConfig(**kwargs)
    config.evaluations_file = get_path(config.data_dir, config.evaluations_file)
    config.agents_evaluations_file = get_path(
        config.data_dir, config.agents_evaluations_file, check_exists=False
    )
    ranker = get_agent_ranker(AgentRankerTypes.ELO, config=config)
    queries = load_answer_evaluations_from_csv(config.evaluations_file)

    ranker.run(queries=queries)
