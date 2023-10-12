import csv
import json
import os
import random
import re
from collections import deque
from typing import Dict, List, Tuple

from loguru import logger
from rich import print
from rich.progress import track
from tenacity import RetryError

from auto_eval.answer_evaluators import AnswerEvaluator, AnswerEvaluatorFactory


@AnswerEvaluatorFactory.register("elo")
class EloEvaluatorWithReasoning(AnswerEvaluator):
    def __init__(self, k: int = 100, bidirectional: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.bidirectional = bidirectional
        self.pattern = re.compile(r"\[\[([^]]+)]].*$(?:(?!\[\[).)*", re.DOTALL)
        self.score_map = {"A": 1, "B": 0, "C": 0.5}
        self.initial_score = 1000
        self.elo_k = 32

    def prepare(self):
        """Prepare the evaluator for running by creating all prompts to be used."""
        self.prompts = self.__create_all_prompts()

    def run(self):
        unparsed_answers = 0
        skip_tuples = set()
        if os.path.exists(self.output_file) and not self.force:
            for line in open(self.output_file):
                data = json.loads(line)
                qid = data["qid"]
                agent_a = data["agent_a"]
                agent_b = data["agent_b"]
                skip_tuples.add((qid, agent_a, agent_b))
            logger.info(f"Skipping {len(skip_tuples)} games...")
        print(f"Running {len(self.prompts) - len(skip_tuples)} games...")

        for p in track(self.prompts, description="Evaluating games"):
            qid = p["qid"]
            agent_a = p["agent_a"]
            agent_b = p["agent_b"]
            prompt_str = p["prompt"]
            logger.debug(f"Running query id {qid} agent {agent_a} vs {agent_b}")
            try:
                gpt_answer = self.openai_client(prompt_str)
            except RetryError:
                logger.warning(
                    f"Failed fetching answer for {qid}, {agent_a}, {agent_b}"
                )
                continue
            logger.debug(gpt_answer)
            try:
                relevant = self.__extract_relevant(gpt_answer)
            except ValueError:
                unparsed_answers += 1
                logger.warning(
                    f"Failed extracting answer for {qid}, {agent_a}, {agent_b}."
                    "Probably not enough tokens in the answer."
                    f"Full answer:\n{gpt_answer}",
                )
                continue
            if self.print:
                print(
                    f"[not bold white][{qid}][/not bold white] "
                    f"[bold blue] {agent_a:<18} [/bold blue] 🆚  "
                    f"[bold red] {agent_b}[/bold red]"
                )
                if relevant == "A":
                    parser_ans = gpt_answer.replace("[[A]]", "[bold blue]A[/bold blue]")
                elif relevant == "B":
                    parser_ans = gpt_answer.replace("[[B]]", "[bold red]B[/bold red]")
                else:
                    parser_ans = gpt_answer.replace(
                        "[[C]]", "[bold purple]C[/bold purple]"
                    )
                print("[bold white] Evaluator answer: [/bold white]")
                print(parser_ans)
            with open(self.output_file, "a") as f:
                json.dump(
                    {
                        "qid": qid,
                        "agent_a": agent_a,
                        "agent_b": agent_b,
                        "prompt": prompt_str,
                        "answer": gpt_answer,
                        "relevant": relevant,
                    },
                    f,
                )

        self.games, self.players = self.__get_elo_scores()
        print(f":check_mark: Done!")
        print(f"Unparsed answers: {unparsed_answers}")
        print(f"Total evaluations: {len(self.prompts) - unparsed_answers}")

    def evaluate(self):
        """Compute score for each agent"""
        while self.games:
            self.__play_one_game()

    def print_ranking(self):
        for player, rating in sorted(
            self.get_player_ratings().items(), key=lambda x: x[1], reverse=True
        ):
            if self.print:
                print(f"[bold white]{player:<15}[/bold white]: [{rating:.1f<6}")

    def __expected_score(self, rating1, rating2):
        return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

    def __update_rating(self, rating, expected_score, actual_score):
        return rating + self.k * (actual_score - expected_score)

    def get_player_ratings(self):
        return self.players

    def __play_one_game(self):
        player1, player2, result = self.games.pop()
        player1_rating = self.players[player1]
        player2_rating = self.players[player2]
        expected_score = self.__expected_score(player1_rating, player2_rating)

        self.players[player1] = self.__update_rating(
            player1_rating, expected_score, result
        )
        self.players[player2] = self.__update_rating(
            player2_rating, 1 - expected_score, 1 - result
        )

    def __get_elo_scores(self) -> Tuple[List[Tuple[str, str, float]], Dict[str, float]]:
        games = []
        players = {}
        for line in open(self.output_file):
            data = json.loads(line)
            agent_a = data["agent_a"]
            agent_b = data["agent_b"]
            relevant = data["relevant"]
            score = self.score_map[relevant]
            games.append((agent_a, agent_b, score))
            if agent_a not in players:
                players[agent_a] = self.initial_score
            if agent_b not in players:
                players[agent_b] = self.initial_score
        random.shuffle(games)
        return games, players

    def __genereate_random_games(self) -> List[Tuple[str, str]]:
        """Creates all prompts necessary for running the evaluator"""
        total_agents = len(self.agents)
        _agents = list(self.agents)
        if self.k == -1:
            logger.info("Creating all possible games...")
            pairs = []
            for i in range(total_agents):
                for j in range(i + 1, total_agents):
                    pairs.append((_agents[i], _agents[j]))
            random.shuffle(pairs)
            return pairs
        rounds_per_agent = self.k // total_agents
        leftover_rounds = self.k % total_agents

        extra_samples_1 = random.sample(_agents, k=leftover_rounds)
        extra_samples_2 = random.sample(_agents, k=leftover_rounds)

        first_agents = _agents * rounds_per_agent + extra_samples_1
        second_agents = _agents * rounds_per_agent + extra_samples_2

        # Shuffle both lists
        random.shuffle(first_agents)
        random.shuffle(second_agents)

        # use deque to pop from the left
        first_agents = deque(first_agents)
        second_agents = deque(second_agents)

        used_pairs = set()  # avoid re-using pairs
        pairs = []
        while first_agents and len(pairs) < self.k:
            agent_a = first_agents.popleft()

            for _ in range(len(second_agents)):
                agent_b = second_agents.popleft()
                if agent_b != agent_a and (agent_a, agent_b) not in used_pairs:
                    used_pairs.add((agent_a, agent_b))
                    pairs.append((agent_a, agent_b))
                    logger.debug(f"Created game {agent_a} vs {agent_b}")
                    if self.bidirectional:
                        pairs.append((agent_b, agent_a))
                        used_pairs.add((agent_b, agent_a))
                        logger.debug(f"Created game {agent_b} vs {agent_a}")
                    second_agents.append(agent_b)
                    break
                second_agents.append(agent_b)
        logger.info(f"Created {len(pairs)} games")
        return pairs

    def __create_all_prompts(self) -> List[Dict[str, str]]:
        prompts = []
        random_pairs = self.__genereate_random_games()
        for qid in track(self.answers):
            query = self.queries[qid]
            for a, b in random_pairs:
                if a not in self.answers[qid] or b not in self.answers[qid]:
                    continue
                reasonings = "\n".join(
                    list(
                        [
                            " ".join([f"[{a}]", b])
                            for (a, b) in self.reasonings[qid].items()
                        ]
                    )
                )
                ans_a = self.answers[qid][a].strip()
                ans_b = self.answers[qid][b].strip()
                prompt = self.prompt.format(
                    query=query, documents=reasonings, answer_a=ans_a, answer_b=ans_b
                )
                prompts.append(
                    {"qid": qid, "agent_a": a, "agent_b": b, "prompt": prompt}
                )
        return prompts

    def ___check_validity(self):
        total_agents = len(self.agents)
        if total_agents < 2:
            raise ValueError(f"Need at least 2 agents, found {total_agents}")
        logger.info(f"Loaded {total_agents} agents")
        if (total_agents * (total_agents - 1)) < self.k:
            possible_games = total_agents * (total_agents - 1)
            logger.warning(
                f"Requested {self.k} games but only {possible_games} are possible"
            )
            logger.warning(f"Will create {possible_games} games per query instead")
            self.k = possible_games

    def __extract_relevant(self, answer: str) -> str:
        """Extracts the relevant part of an answer."""
        match_ans = self.pattern.search(answer)
        if not match_ans:
            raise ValueError(f"Could not find answer in {answer}")
        answer = match_ans.group(1)
        if answer not in ["A", "B", "C"]:
            raise ValueError(f"Unknown answer: {answer}")
        return answer
