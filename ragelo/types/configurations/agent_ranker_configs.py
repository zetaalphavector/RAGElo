from typing import List

from pydantic import Field

from ragelo.types.configurations.base_configs import BaseConfig


class AgentRankerConfig(BaseConfig):
    ranker_name: str = Field(default="elo", description="The name of the agent ranker")
    output_columns_agent_ranker: List[str] = Field(
        default=["agent", "score"],
        description="The columns to output in the CSV file",
    )
    agents_evaluations_file: str = Field(
        default="agents_evaluations.csv",
        description="Path to the agents evaluations file",
    )
    verbose: bool = Field(default=True, description="Whether to print the ranking")
    evaluations_file: str = Field(
        default="pairwise_answers_evaluations.csv",
        description="Path to the pairwise evaluations file",
    )


class EloAgentRankerConfig(AgentRankerConfig):
    elo_k: int = Field(
        default=32, description="The K factor for the Elo ranking algorithm"
    )
    initial_score: int = Field(
        default=1000, description="The initial score for each agent"
    )
    agents_evaluations_file: str = Field(
        default="elo_ranking.csv", description="Path to the output file"
    )
    rounds: int = Field(default=10, description="The number of rounds to play")
