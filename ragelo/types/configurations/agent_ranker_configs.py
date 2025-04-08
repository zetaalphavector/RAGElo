from __future__ import annotations

from pydantic import Field

from ragelo.types.configurations.base_configs import BaseConfig


class AgentRankerConfig(BaseConfig):
    ranker_name: str = Field(default="elo", description="The name of the agent ranker")
    output_columns_agent_ranker: list[str] = Field(
        default=["agent", "score"],
        description="The columns to output in the CSV file",
    )
    agents_evaluations_file: str = Field(
        default="agents_evaluations.csv",
        description="Path to the agents evaluations file",
    )
    verbose: bool = Field(default=False, description="Whether to print each game played")
    evaluations_file: str = Field(
        default="pairwise_answers_evaluations.csv",
        description="Path to the pairwise evaluations file",
    )


class EloAgentRankerConfig(AgentRankerConfig):
    elo_k: int = Field(default=32, description="The K factor for the Elo ranking algorithm")
    initial_score: int = Field(default=1000, description="The initial score for each agent")
    agents_evaluations_file: str = Field(default="elo_ranking.csv", description="Path to the output file")
    tournaments: int = Field(default=10, description="The number of Elo tournaments to play")
    score_mapping: dict[str, float] = Field(
        default={"A": 1, "B": 0, "C": 0.5},
        description="The mapping of the evaluation answers to scores",
    )
