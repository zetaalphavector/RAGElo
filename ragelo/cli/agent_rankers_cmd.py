import os

import typer

from ragelo.agent_rankers import AgentRankerFactory
from ragelo.cli.args import get_params_from_function
from ragelo.types import EloAgentRankerConfig

typer.main.get_params_from_function = get_params_from_function  # type: ignore
app = typer.Typer()


@app.command()
def EloRanker(config: EloAgentRankerConfig = EloAgentRankerConfig(), **kwargs):
    """Evaluate the performance of multiple agents using an Elo ranking system."""
    if kwargs["output_file"] is None:
        kwargs["output_file"] = os.path.join(
            kwargs.get("data_path", ""), "elo_ranking.csv"
        )

    config = EloAgentRankerConfig(**kwargs)
    ranker = AgentRankerFactory.create("elo", config)
    ranker.run()
