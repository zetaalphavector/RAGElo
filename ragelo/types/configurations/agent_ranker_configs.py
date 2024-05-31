from typing import List, Optional

from pydantic import Field

from ragelo.types.configurations.base_configs import BaseConfig


class AgentRankerConfig(BaseConfig):
    ranker_name: str = Field(default="elo", description="The name of the agent ranker")
    output_columns: List[str] = Field(
        default=["agent", "score"],
        description="The columns to output in the CSV file",
    )
    output_file: Optional[str] = Field(
        default=None, description="Path to the output file"
    )
    verbose: bool = Field(default=True, description="Whether to print the ranking")
    evaluations_file: str = Field(
        default="pairwise_answer_evaluations.csv",
        description="Path to the pairwise evaluations file",
    )


class EloAgentRankerConfig(AgentRankerConfig):
    k: int = Field(default=32, description="The K factor for the Elo ranking algorithm")
    initial_score: int = Field(
        default=1000, description="The initial score for each agent"
    )
    output_file: str = Field(
        default="elo_ranking.csv", description="Path to the output file"
    )
    rounds: int = Field(default=10, description="The number of rounds to play")
