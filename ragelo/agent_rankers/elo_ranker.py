from __future__ import annotations

import math
import random
from typing import Any, Awaitable, Callable, Optional

import numpy as np
from tqdm.auto import tqdm

from ragelo.agent_rankers.base_agent_ranker import AgentRanker, AgentRankerFactory
from ragelo.evaluators.answer_evaluators import BaseAnswerEvaluator
from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator
from ragelo.logger import logger
from ragelo.types.configurations import EloAgentRankerConfig
from ragelo.types.evaluables import AgentAnswer, Document, PairwiseGame
from ragelo.types.experiment import Experiment
from ragelo.types.query import Query
from ragelo.types.results import AnswerEvaluatorResult, EloTournamentResult
from ragelo.types.types import AgentRankerTypes


@AgentRankerFactory.register(AgentRankerTypes.ELO)
class EloRanker(AgentRanker):
    name: str = "Elo Agent Ranker"
    config: EloAgentRankerConfig

    def __init__(self, config: EloAgentRankerConfig):
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
        self.games: list[tuple[str, str, str]] = []
        self.evaluations: list[tuple[str, str, str, str]] = []

    def run(self, experiment: Experiment) -> EloTournamentResult:
        """Compute score for each agent"""
        self.evaluations = self._flatten_evaluations(experiment)
        agent_scores: dict[str, list[int]] = {}
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
        experiment.add_evaluation(result, should_print=True)
        return result

    def get_agents_ratings(self):
        return self.agents_scores

    def get_ranked_agents(self) -> list[tuple[str, float]]:
        ranking = sorted(self.get_agents_ratings().items(), key=lambda x: x[1], reverse=True)
        return [(agent, rating) for agent, rating in ranking]

    def run_tournament(self) -> dict[str, int]:
        agents_scores: dict[str, int] = {}
        games: list[tuple[str, str, float]] = []
        for _, agent_a, agent_b, score in self.evaluations:
            score_val = self.score_map[score]
            games.append((agent_a, agent_b, score_val))
        random.shuffle(games)
        for agent_a, agent_b, score_val in games:
            if self.config.verbose:
                logger.info(f"Game: {agent_a} vs {agent_b} -> {score_val}")
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

            expected_score = 1 / (1 + 10 ** ((agent_a_rating - agent_b_rating) / 400))
            agents_scores[agent_a] = int(agent_a_rating + self.k * (score_val - expected_score))
            agents_scores[agent_b] = int(agent_b_rating + self.k * ((1 - score_val) - (1 - expected_score)))
            self.total_games += 1
            self.games_played[agent_a] = self.games_played.get(agent_a, 0) + 1
            self.games_played[agent_b] = self.games_played.get(agent_b, 0) + 1

        self.computed = True
        return agents_scores

    def get_agent_losses(self, agent: str) -> list[str, str]:
        """For a given agent, returns a list of tuples(qid, agent) with the query ids and the agents that the agent lost to"""
        lost_games = []
        for qid, agent_a, agent_b, winner in self.games:
            if agent_b == agent and winner == "A":
                lost_games.append((qid, agent_a))
            elif agent_a == agent and winner == "B":
                lost_games.append((qid, agent_b))
        return lost_games

    async def run_single_game(
        self,
        query: Query,
        agent_a: str,
        agent_b: str,
        answer_evaluator: BaseAnswerEvaluator,
        retrieval_evaluator: Optional[BaseRetrievalEvaluator] = None,
        experiment: Optional[Experiment] = None,
    ) -> tuple[AnswerEvaluatorResult, AnswerEvaluatorResult]:
        """Run a single game between two agents"""
        assert answer_evaluator.config.pairwise, "Answer evaluator must be pairwise"
        assert agent_a in query.answers, f"Agent {agent_a} not found in query"
        assert agent_b in query.answers, f"Agent {agent_b} not found in query"

        answer_a = query.answers[agent_a]
        answer_b = query.answers[agent_b]

        if retrieval_evaluator:
            retrieval_evaluator.evaluate_all_evaluables(query, n_threads=10)
            if experiment:
                experiment.save()
        evaluation = await answer_evaluator.evaluate_async(
            (
                query,
                PairwiseGame(
                    qid=query.qid,
                    agent_a_answer=answer_a,
                    agent_b_answer=answer_b,
                ),
            )
        )
        winner = evaluation.answer.winner
        score_val = self.score_map[winner]
        # update dictionaries and rankings
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
        return evaluation

    def update_rankings(self, agent_a: str, agent_b: str, score_val: float) -> tuple[int, int]:
        agent_a_rating = self.agents_scores.get(agent_a, self.initial_score)
        agent_b_rating = self.agents_scores.get(agent_b, self.initial_score)

        expected_score = 1 / (1 + 10 ** ((agent_a_rating - agent_b_rating) / 400))
        self.agents_scores[agent_a] = int(agent_a_rating + self.k * (score_val - expected_score))
        self.agents_scores[agent_b] = int(agent_b_rating + self.k * ((1 - score_val) - (1 - expected_score)))
        self.games_played[agent_a] = self.games_played.get(agent_a, 0) + 1
        self.games_played[agent_b] = self.games_played.get(agent_b, 0) + 1
        return self.agents_scores[agent_a], self.agents_scores[agent_b]

    def add_new_agent(self, agent: str):
        self.agents_scores[agent] = self.initial_score
        self.games_played[agent] = 0
        self.wins[agent] = 0
        self.losses[agent] = 0
        self.ties[agent] = 0
        self.total_games = 0
        return self.agents_scores[agent]

    def _expected_score(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

    def _opponent_info_value(self, new_rating: float, opponent: str) -> float:
        rb = self.agents_scores.get(opponent, self.initial_score)
        p = self._expected_score(new_rating, rb)  # info ~ p(1-p), max at 0.5
        opp_uncert = 1.0 / math.sqrt(self.games_played.get(opponent, 0) + 1)
        return (p * (1.0 - p)) * opp_uncert

    def _question_entropy_scores(self) -> dict[str, float]:
        per_q_counts: dict[str, dict[str, int]] = {}
        for qid, _a, _b, w in self.evaluations:
            d = per_q_counts.setdefault(qid, {"A": 0, "B": 0, "C": 0})
            if w in d:
                d[w] += 1
        scores: dict[str, float] = {}
        for qid, d in per_q_counts.items():
            n = sum(d.values())
            if n == 0:
                scores[qid] = 0.0
                continue
            # Shannon entropy over A/B/C, penalize tie-heavy questions
            probs = [c / n for c in d.values() if c > 0]
            entropy = -sum(p * math.log(p + 1e-12) for p in probs)  # nats
            tie_penalty = 1.0 - (d.get("C", 0) / n)
            scores[qid] = entropy * tie_penalty
        return scores

    def _wilson_ci(self, k: float, n: int, z: float) -> tuple[float, float]:
        if n == 0:
            return (0.0, 1.0)
        phat = k / n
        denom = 1.0 + (z * z) / (2.0 * n)
        center = (phat + (z * z) / (2.0 * n)) / denom
        radius = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4.0 * n)) / n)) / denom
        return (max(0.0, center - radius), min(1.0, center + radius))

    async def add_agent_without_games(
        self,
        experiment: Experiment,
        new_agent: str,
        answer_evaluator: BaseAnswerEvaluator,
        retrieval_evaluator: Optional[object] = None,
        agent_callable: Optional[Callable[[str, str], Awaitable[tuple[str, list[Any]]]]] = None,
    ) -> EloTournamentResult:
        """Estimate Elo for a brand-new agent with minimal games.

        Strategy:
        - If no opponents exist, return initial score.
        - If exactly one opponent exists, play a few high-entropy queries against it.
        - Otherwise, pick up to a few informative opponents (based on expected score uncertainty
          and opponent games played), and high-entropy questions. Stop early when Wilson CI over
          the new agent's observed score is sufficiently tight or when max game budget is reached.
        """
        assert answer_evaluator.config.pairwise, "Answer evaluator must be pairwise"
        # if agent_callable is None:
        # raise ValueError("agent_callable must be provided to generate answers and documents for the new agent")

        # Ensure internal state for the new agent exists
        logger.info(f"Adding new agent: {new_agent}")
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
            if self.config.verbose:
                logger.info("No existing agents found. Returning initial score for the new agent.")
            return EloTournamentResult(
                agents=[new_agent],
                scores={new_agent: float(self.agents_scores.get(new_agent, self.initial_score))},
                games_played={new_agent: 0},
                wins={new_agent: 0},
                loses={new_agent: 0},
            )

        # Prepare evaluations summary to score questions by entropy (if any exist)
        self.evaluations = self._flatten_evaluations(experiment)
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
        max_games_budget = 10
        min_games_before_ci_check = 3
        target_ci_width = 0.30  # 95% CI half-width target for "somewhat reliable"

        # Select opponents
        if len(existing_agents) == 1:
            selected_opponents = list(existing_agents)
        else:
            # Rank opponents by information value for the current new agent rating
            info_values = [
                (opp, self._opponent_info_value(self.agents_scores.get(new_agent, self.initial_score), opp))
                for opp in existing_agents
            ]
            info_values.sort(key=lambda x: x[1], reverse=True)
            selected_opponents = [opp for opp, _ in info_values[: min(3, len(info_values))]]

        # Prepare round-robin iterators over informative queries per selected opponent
        per_opp_indices: dict[str, int] = {opp: 0 for opp in selected_opponents}

        # Tracking success proportion for Wilson CI (treat tie as 0.5 success)
        new_agent_wins_before = self.wins.get(new_agent, 0)
        new_agent_ties_before = self.ties.get(new_agent, 0)
        new_agent_games_before = self.games_played.get(new_agent, 0)
        observed_success_sum = 0.0
        observed_games = 0

        # Helper to check CI and early stop
        def should_stop() -> bool:
            if observed_games < min_games_before_ci_check:
                return False
            low, high = self._wilson_ci(observed_success_sum, observed_games, 1.96)
            return (high - low) <= target_ci_width

        # Avoid repeating same (qid, opp) games
        played_pairs: set[tuple[str, str]] = set()

        # Main loop: pick next opponent and next informative query
        pbar = tqdm(total=max_games_budget, desc=f"Playing games for {new_agent}", leave=False, ncols=120)
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
                    query = queries_by_id[qid]
                    # Ensure the new agent has an answer; if not, generate it
                    if new_agent not in query.answers:
                        try:
                            answer_text, docs = await agent_callable(query.qid, query.query)
                        except Exception as e:
                            logger.warning(f"Failed to get answer/docs for new agent on qid={query.qid}: {e}")
                            continue
                        # Add retrieved docs (if any)
                        if docs:
                            for d in docs:
                                try:
                                    if isinstance(d, Document):
                                        experiment.add_retrieved_doc(d, exist_ok=True, agent=new_agent)
                                    else:
                                        # Fallback: attempt to assemble from string
                                        experiment.add_retrieved_doc(
                                            str(d), query_id=query.qid, doc_id=str(d), agent=new_agent, exist_ok=True
                                        )
                                except Exception as e:
                                    logger.debug(f"Skipping doc add for qid={query.qid}: {e}")
                        # Add the new agent answer
                        try:
                            experiment.add_agent_answer(
                                AgentAnswer(qid=query.qid, agent=new_agent, text=str(answer_text)),
                                exist_ok=True,
                            )
                        except Exception as e:
                            logger.warning(f"Failed to add new agent answer for qid={query.qid}: {e}")
                            continue

                    # Run a single pairwise game (new_agent vs opp) on this query
                    wins_before = self.wins.get(new_agent, 0)
                    ties_before = self.ties.get(new_agent, 0)
                    try:
                        evaluation = await self.run_single_game(
                            query=query,
                            agent_a=new_agent,
                            agent_b=opp,
                            answer_evaluator=answer_evaluator,
                            retrieval_evaluator=retrieval_evaluator,  # type: ignore
                            experiment=experiment,
                        )
                        # Persist at least one orientation for caching
                        try:
                            experiment.add_evaluation(evaluation, should_save=True, should_print=False, exist_ok=True)
                        except Exception:
                            pass
                    except Exception as e:
                        logger.warning(f"Failed to run game on qid={query.qid} between {new_agent} and {opp}: {e}")
                        pbar.update(1)
                        continue
                    pbar.update(1)
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
        # Return the current rating estimate for the new agent
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
