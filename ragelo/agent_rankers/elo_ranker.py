from __future__ import annotations

import logging
import random
from typing import Awaitable, Callable

import numpy as np

from ragelo.agent_rankers.base_agent_ranker import AgentRanker, AgentRankerFactory
from ragelo.evaluators.answer_evaluators.pairwise_evaluator import PairwiseAnswerEvaluator
from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import BaseRetrievalEvaluator
from ragelo.types import (
    AgentAnswer,
    EloAgentRankerConfig,
    EloTournamentResult,
    Experiment,
    PairwiseGameEvaluatorResult,
    Query,
)
from ragelo.types.types import AgentRankerTypes
from ragelo.utils import get_pbar

logger = logging.getLogger(__name__)


@AgentRankerFactory.register(AgentRankerTypes.ELO)
class EloRanker(AgentRanker):
    name: str = "Elo Agent Ranker"
    config: EloAgentRankerConfig

    def __init__(
        self,
        config: EloAgentRankerConfig,
    ):
        super().__init__(config)
        self.score_map = config.score_mapping

        self.agents_scores: dict[str, float] = {}
        self.wins: dict[str, int] = {}
        self.losses: dict[str, int] = {}
        self.ties: dict[str, int] = {}
        self.total_games: int = 0
        self.games_played: dict[str, int] = {}
        self.computed: bool = False
        self.initial_score: int = self.config.initial_score
        self.k: int = self.config.elo_k
        self.std_dev: dict[str, float] = {}
        self.games: list[tuple[str, str, str, str]] = []

    def add_new_agent(self, agent: str):
        """Add a new agent to the rankings.

        Args:
            agent: The name of the agent to add.
        """
        if agent not in self.agents_scores:
            self.agents_scores[agent] = self.initial_score
            self.games_played[agent] = 0
            self.wins[agent] = 0
            self.losses[agent] = 0
            self.ties[agent] = 0
            self.std_dev[agent] = 0

    def run(self, experiment: Experiment, evaluator: PairwiseAnswerEvaluator | None = None) -> EloTournamentResult:
        """Run an Elo-based tournament with all agents in an experiment.

        Args:
            experiment: The experiment to run.
        """
        self.evaluations = self._flatten_evaluations(
            experiment,
            evaluator_name=evaluator.config.evaluator_name if evaluator else None,
        )
        agent_scores: dict[str, list[float]] = {}
        for _ in range(self.config.tournaments):
            results = self.run_tournament()
            for agent, score in results.items():
                agent_scores[agent] = agent_scores.get(agent, []) + [score]
        for a in agent_scores:
            self.std_dev[a] = float(np.std(agent_scores[a]))
            self.agents_scores[a] = float(np.mean(agent_scores[a]))

        result = EloTournamentResult(
            agents=list(self.agents_scores.keys()),
            scores=self.agents_scores,
            games_played=self.games_played,
            wins=self.wins,
            loses=self.losses,
            ties=self.ties,
            std_dev=self.std_dev,
            total_games=self.total_games,
            total_tournaments=self.config.tournaments,
        )
        experiment.add_evaluation(None, result, should_print=self.config.render)
        return result

    def get_agents_ratings(self):
        """Get the ratings of all agents."""
        return self.agents_scores

    def get_ranked_agents(self) -> list[tuple[str, float]]:
        """Get the ranked agents by their ratings."""
        ranking = sorted(self.get_agents_ratings().items(), key=lambda x: x[1], reverse=True)
        return [(agent, rating) for agent, rating in ranking]

    def run_tournament(self) -> dict[str, float]:
        """Run a single tournament."""
        agents_scores: dict[str, float] = {}
        games: list[tuple[str, str, float]] = []
        for _, agent_a, agent_b, score in self.evaluations:
            score_val = self.score_map[score]
            games.append((agent_a, agent_b, score_val))
        random.shuffle(games)
        for agent_a, agent_b, score_val in games:
            logger.debug(f"Game: {agent_a} vs {agent_b} -> {score_val}")
            if score_val == 1:
                self.wins[agent_a] = self.wins.get(agent_a, 0) + 1
                self.losses[agent_b] = self.losses.get(agent_b, 0) + 1
            elif score_val == 0:
                self.wins[agent_b] = self.wins.get(agent_b, 0) + 1
                self.losses[agent_a] = self.losses.get(agent_a, 0) + 1
            else:
                self.ties[agent_a] = self.ties.get(agent_a, 0) + 1
                self.ties[agent_b] = self.ties.get(agent_b, 0) + 1
            agent_a_rating = agents_scores.get(agent_a, self.initial_score)
            agent_b_rating = agents_scores.get(agent_b, self.initial_score)

            expected_score = self._expected_score(agent_a_rating, agent_b_rating)
            agents_scores[agent_a] = agent_a_rating + self.k * (score_val - expected_score)
            agents_scores[agent_b] = agent_b_rating + self.k * ((1 - score_val) - (1 - expected_score))
            self.total_games += 1
            self.games_played[agent_a] = self.games_played.get(agent_a, 0) + 1
            self.games_played[agent_b] = self.games_played.get(agent_b, 0) + 1

        self.computed = True
        return agents_scores

    async def run_single_game(
        self,
        query: Query,
        agent_a: str,
        agent_b: str,
        evaluator: PairwiseAnswerEvaluator,
        retrieval_evaluator: BaseRetrievalEvaluator | None = None,
        experiment: Experiment | None = None,
    ) -> tuple[float, float]:
        """Run a single game between two agents and update their rankings according to the Elo algorithm.

        Args:
            query: The query to run the game on.
            agent_a: The name of the first agent.
            agent_b: The name of the second agent.
            evaluator: The evaluator to use for the game.
            retrieval_evaluator: The retrieval evaluator to use for the game.
            experiment: The experiment to save the game to.

        Returns:
            The ratings of the two agents after the game.
        """
        assert agent_a in query.answers, f"Agent {agent_a} not found in query"
        assert agent_b in query.answers, f"Agent {agent_b} not found in query"
        game = query.add_pairwise_game(agent_a, agent_b, exist_ok=True)

        if retrieval_evaluator:
            retrieval_evaluator.evaluate_all_evaluables(query, n_threads=10)
            if experiment:
                experiment.save()
        evaluation = await evaluator.evaluate_async((query, game))
        assert isinstance(
            evaluation, PairwiseGameEvaluatorResult
        ), f"Evaluation {evaluation} is not a PairwiseGameEvaluatorResult"
        if experiment:
            experiment.add_evaluation(
                (query, game),
                evaluation,
                exist_ok=True,
                should_print=self.config.render,
            )
        winner = evaluation.winner
        score_val = self.score_map[winner]
        agent_a = game.agent_a_answer.agent
        agent_b = game.agent_b_answer.agent
        if winner == "A":
            self.wins[agent_a] = self.wins.get(agent_a, 0) + 1
            self.losses[agent_b] = self.losses.get(agent_b, 0) + 1
        elif winner == "B":
            self.wins[agent_b] = self.wins.get(agent_b, 0) + 1
            self.losses[agent_a] = self.losses.get(agent_a, 0) + 1
        else:
            self.ties[agent_a] = self.ties.get(agent_a, 0) + 1
            self.ties[agent_b] = self.ties.get(agent_b, 0) + 1
        self.games.append((query.qid, agent_a, agent_b, winner))
        self.update_rankings(agent_a, agent_b, score_val)
        return self.agents_scores[agent_a], self.agents_scores[agent_b]

    def update_rankings(self, agent_a: str, agent_b: str, score_val: float) -> tuple[float, float]:
        """Update the rankings of two agents according to the Elo algorithm.

        Args:
            agent_a: The name of the first agent.
            agent_b: The name of the second agent.
            score_val: The score of the game.

        Returns:
            The ratings of the two agents after the update.
        """
        agent_a_rating = self.agents_scores.get(agent_a, self.initial_score)
        agent_b_rating = self.agents_scores.get(agent_b, self.initial_score)

        expected_score = self._expected_score(agent_a_rating, agent_b_rating)
        self.agents_scores[agent_a] = agent_a_rating + self.k * (score_val - expected_score)
        self.agents_scores[agent_b] = agent_b_rating + self.k * ((1 - score_val) - (1 - expected_score))
        self.games_played[agent_a] = self.games_played.get(agent_a, 0) + 1
        self.games_played[agent_b] = self.games_played.get(agent_b, 0) + 1
        return self.agents_scores[agent_a], self.agents_scores[agent_b]

    def _expected_score(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def _question_entropy_scores(self) -> dict[str, float]:
        per_q_counts: dict[str, dict[str, int]] = {}
        for qid, _a, _b, w in self.evaluations:
            d = per_q_counts.setdefault(qid, {"A": 0, "B": 0, "C": 0})
            if w in d:
                d[w] += 1
        entropies: dict[str, float] = {}
        for qid, d in per_q_counts.items():
            n = sum(d.values())
            if n == 0:
                entropies[qid] = 0.0
                continue
            # Shannon entropy over A/B/C, penalize tie-heavy questions
            probs = [c / n for c in d.values() if c > 0]
            entropy = -sum(p * np.log(p + 1e-12) for p in probs)  # nats
            tie_penalty = 1.0 - (d.get("C", 0) / n)
            entropies[qid] = entropy * tie_penalty
        return entropies

    def _opponent_info_value(self, new_rating: float, opponent: str) -> float:
        rb = self.agents_scores.get(opponent, self.initial_score)
        p = self._expected_score(new_rating, rb)  # info ~ p(1-p), max at 0.5
        opp_uncert = 1.0 / np.sqrt(self.games_played.get(opponent, 0) + 1)
        return (p * (1.0 - p)) * opp_uncert

    def _wilson_ci(self, k: float, n: int, z: float) -> tuple[float, float]:
        if n == 0:
            return (0.0, 1.0)
        phat = k / n
        denom = 1.0 + (z * z) / (2.0 * n)
        center = (phat + (z * z) / (2.0 * n)) / denom
        radius = (z * np.sqrt((phat * (1 - phat) + (z * z) / (4.0 * n)) / n)) / denom
        return (max(0.0, center - radius), min(1.0, center + radius))

    async def add_agent_to_tournament(
        self,
        experiment: Experiment,
        new_agent: str,
        evaluator: PairwiseAnswerEvaluator,
        retrieval_evaluator: BaseRetrievalEvaluator | None = None,
        agent_callable: Callable[[str, str], Awaitable[AgentAnswer]] | None = None,
    ) -> EloTournamentResult:
        """Estimate Elo for a brand-new agent with minimal games.
        Args:
            experiment: The experiment to add the agent to.
            new_agent: The name of the new agent.
            evaluator: The evaluator to use for the agent.
            retrieval_evaluator: The retrieval evaluator to use for the agent.
            agent_callable: A callable that returns the answers of the agent for a given query.
        Returns:
            The Elo tournament result.

        Strategy:
        - If no opponents exist, return initial score.
        - If exactly one opponent exists, play a few high-entropy queries against it.
        - Otherwise, pick up to a few informative opponents (based on expected score uncertainty
          and opponent games played), and high-entropy questions. Stop early when Wilson CI over
          the new agent's observed score is sufficiently tight or when max game budget is reached.
        """
        logger.info(f"Adding new agent: {new_agent} to tournament")
        if new_agent not in self.agents_scores:
            self.add_new_agent(new_agent)
        # Discover existing agents from the experiment (excluding the new one)
        existing_agents: set[str] = set()
        for q in experiment:
            existing_agents.update(q.answers.keys())
        if new_agent in existing_agents:
            existing_agents.remove(new_agent)
        # Edge case: No opponents available
        if len(existing_agents) == 0:
            return EloTournamentResult(
                agents=[new_agent],
                scores={new_agent: float(self.agents_scores.get(new_agent, self.initial_score))},
                games_played={new_agent: 0},
                wins={new_agent: 0},
                loses={new_agent: 0},
                ties={new_agent: 0},
                std_dev={new_agent: 0},
                total_games=0,
                total_tournaments=0,
            )
        # Prepare evaluations summary to score questions by entropy (if any exist)
        self.evaluations = self._flatten_evaluations(experiment, evaluator_name=evaluator.config.evaluator_name)
        question_entropy = self._question_entropy_scores() if len(self.evaluations) > 0 else {}

        # Build per-opponent list of queries they have answered
        queries_by_id: dict[str, Query] = {q.qid: q for q in experiment}
        opponent_answered_qids: dict[str, list[str]] = {}
        for opp in existing_agents:
            qids = [qid for qid, q in queries_by_id.items() if opp in q.answers]
            # Sort qids by entropy (desc) then by number of answers in that query (desc) as a fallback signal
            qids.sort(
                key=lambda qid: (
                    question_entropy.get(qid, 0.0),
                    len(queries_by_id[qid].answers),
                ),
                reverse=True,
            )
            opponent_answered_qids[opp] = qids

        # Edge case: Exactly one opponent
        max_games_budget = 20
        min_games_before_ci_check = 3
        target_ci_width = 0.30  # 95% CI half-width target for "somewhat reliable"

        # Select opponents
        if len(existing_agents) == 1:
            selected_opponents = list(existing_agents)
        else:
            # Rank opponents by information value for the current new agent rating
            info_values = [
                (
                    opp,
                    self._opponent_info_value(self.agents_scores.get(new_agent, self.initial_score), opp),
                )
                for opp in existing_agents
            ]
            info_values.sort(key=lambda x: x[1], reverse=True)
            selected_opponents = [opp for opp, _ in info_values[: min(3, len(info_values))]]

        # Prepare round-robin iterators over informative queries per selected opponent
        per_opp_indices: dict[str, int] = {opp: 0 for opp in selected_opponents}

        # Tracking success proportion for Wilson CI (treat tie as 0.5 success)
        observed_success_sum = 0.0
        observed_games = 0

        # Helper to check CI and early stop
        def should_stop() -> bool:
            if observed_games < min_games_before_ci_check:
                return False
            low, high = self._wilson_ci(observed_success_sum, observed_games, 1.96)
            return (high - low) <= target_ci_width

        played_pairs: set[tuple[str, str]] = set()
        # Main loop: pick next opponent and next informative query
        pbar = get_pbar(
            max_games_budget,
            use_rich=self.config.rich_print,
            desc=f"Playing games for {new_agent}",
            disable=not self.config.use_progress_bar,
        )
        while observed_games < max_games_budget and len(selected_opponents) > 0:
            progressed = False
            for opp in list(selected_opponents):
                qids = opponent_answered_qids.get(opp, [])
                # Advance to next usable query for this opponent
                while per_opp_indices[opp] < len(qids):
                    qid = qids[per_opp_indices[opp]]
                    per_opp_indices[opp] += 1
                    # Skip if this (qid, opp) was already played in this routine
                    if (qid, opp) in played_pairs:
                        continue
                    query = experiment[qid]
                    if new_agent not in query.answers:
                        if not agent_callable:
                            logger.warning(f"Failed to get answer/docs for new agent on qid={query.qid}")
                            continue
                        try:
                            agent_answer = await agent_callable(query.qid, query.query)
                        except Exception as e:
                            logger.warning(f"Failed to get answer/docs for new agent on qid={query.qid}: {e}")
                            continue
                        # Add the new agent answer
                        try:
                            experiment.add_agent_answer(agent_answer, exist_ok=True)
                        except Exception as e:
                            logger.warning(f"Failed to add new agent answer for qid={query.qid}: {e}")
                            continue
                    # Run a single pairwise game (new_agent vs opp) on this query
                    wins_before = self.wins.get(new_agent, 0)
                    ties_before = self.ties.get(new_agent, 0)
                    try:
                        await self.run_single_game(
                            query=query,
                            agent_a=new_agent,
                            agent_b=opp,
                            evaluator=evaluator,
                            retrieval_evaluator=retrieval_evaluator,  # type: ignore
                            experiment=experiment,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to run single game for qid={query.qid}: {e}")
                        pbar.update()
                        continue
                    pbar.update()
                    # Update success tracking for CI
                    wins_after = self.wins.get(new_agent, 0)
                    ties_after = self.ties.get(new_agent, 0)
                    delta_wins = max(0, wins_after - wins_before)
                    delta_ties = max(0, ties_after - ties_before)
                    observed_success_sum += float(delta_wins) + 0.5 * float(delta_ties)
                    observed_games += 1
                    played_pairs.add((qid, opp))
                    progressed = True

                    if should_stop() or observed_games >= max_games_budget:
                        return EloTournamentResult(
                            agents=list(self.agents_scores.keys()),
                            scores=self.agents_scores,
                            games_played=self.games_played,
                            wins=self.wins,
                            loses=self.losses,
                            ties=self.ties,
                            std_dev=self.std_dev,
                            total_games=self.total_games,
                            total_tournaments=self.config.tournaments,
                        )

                # If no more queries left for this opponent, drop it
                if per_opp_indices[opp] >= len(qids):
                    selected_opponents.remove(opp)

            if not progressed:
                # No progress possible (e.g., no queries with opponent answers). Stop.
                break
        pbar.close()
        return EloTournamentResult(
            agents=list(self.agents_scores.keys()),
            scores=self.agents_scores,
            games_played=self.games_played,
            wins=self.wins,
            loses=self.losses,
            ties=self.ties,
            std_dev=self.std_dev,
            total_games=self.total_games,
            total_tournaments=self.config.tournaments,
        )
