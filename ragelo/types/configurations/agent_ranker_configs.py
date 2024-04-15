from typing import Optional

from pydantic import Field

from ragelo.types.configurations.base_configs import BaseConfig


class AgentRankerConfig(BaseConfig):
    evaluations_file: str = Field(
        default="data/evaluations.csv",
        description="Path with the pairwise evaluations of answers",
    )
    output_file: Optional[str] = Field(
        default=None, description="Path to the output file"
    )


class EloAgentRankerConfig(AgentRankerConfig):
    k: int = Field(default=32, description="The K factor for the Elo ranking algorithm")
    initial_score: int = Field(
        default=1000, description="The initial score for each agent"
    )
    output_file: str = Field(
        default="elo_ranking.csv", description="Path to the output file"
    )
