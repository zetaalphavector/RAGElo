import csv
import json
import os
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import List, Tuple

from loguru import logger
from rich import print
from rich.progress import track


class GamesCreator:
    """Creates an uniformily sampled set of pairwise games from a list of answers"""

    def __init__(
        self,
        queries_file: str,
        answers_path: str,
        output_file: str,
        k: int = 100,
        reasonings_file: str | None = None,
        prompt_template: str = "pairwise_prompt",
        bidirectional: bool = False,
        force: bool = False,
        print: bool = False,
    ):
        self.games = []
        self.players = set()
        self.query_to_qid = {}
        self.qid_to_query = {}
        self.reasonings = defaultdict(dict)
        self.answers = defaultdict(dict)
        self.bidirectional = bidirectional
        self.output_file = output_file
        self.k = k
        self.print = print

        if not force and os.path.isfile(output_file):
            raise FileExistsError(
                f"Cowardly refusing to re-create games. "
                "Output file {output_file} already exists."
            )

        self._load_prompt(prompt_template)
        self._read_queries(queries_file)
        if reasonings_file:
            self._read_reasonings(reasonings_file)
        else:
            # TODO: Different prompts may have different requirements.
            raise NotImplementedError("Reasonings file is required")
        self._read_answers(answers_path)

    def create_all_prompts(self):
        """Creates all prompts to be submitted to OpenAI."""
        with open(self.output_file, "w") as f:
            for qid in track(self.answers, description=f"Assembling prompts"):
                query = self.qid_to_query[qid]
                random_pairs = self.create_random_games()
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
                    ans_a = (
                        self.answers[qid][a].strip().replace("CONFIDENTIAL_REPORT/", "")
                    )

                    ans_b = (
                        self.answers[qid][b].strip().replace("CONFIDENTIAL_REPORT/", "")
                    )
                    prompt = self.prompt.format(
                        query=query,
                        documents=reasonings,
                        answer_a=ans_a,
                        answer_b=ans_b,
                    )
                    json.dump(
                        {
                            "qid": qid,
                            "query": query,
                            "agent_a": a,
                            "agent_b": b,
                            "answer_a": ans_a,
                            "answer_b": ans_b,
                            "prompt": prompt,
                        },
                        f,
                    )
                    f.write("\n")
        print(f"Created {self.k} games per query in {self.output_file}")
        print(f"total games: {self.k * len(self.answers)}")

    def create_random_games(self) -> List[Tuple[str, str]]:
        total_players = len(self.players)
        _players = list(self.players)

        if self.k == -1:
            # create all possible pairs
            pairs = []
            for i in range(total_players):
                for j in range(i + 1, total_players):
                    pairs.append((_players[i], _players[j]))
            random.shuffle(pairs)
            return pairs
        rounds_per_player = self.k // total_players
        leftover_rounds = self.k % total_players

        extra_samples_1 = random.sample(_players, k=leftover_rounds)
        extra_samples_2 = random.sample(_players, k=leftover_rounds)

        first_elements = _players * rounds_per_player + extra_samples_1
        second_elements = _players * rounds_per_player + extra_samples_2

        # Shuffle both lists
        random.shuffle(first_elements)
        random.shuffle(second_elements)

        # Use deque to pop from the left
        first_elements = deque(first_elements)
        second_elements = deque(second_elements)

        used_pairs = set()  # avoid re-using pairs
        pairs = []
        while first_elements and len(pairs) < self.k:
            agent_a = first_elements.popleft()

            for _ in range(len(second_elements)):
                agent_b = second_elements.popleft()
                if agent_b != agent_a and (agent_a, agent_b) not in used_pairs:
                    used_pairs.add((agent_a, agent_b))
                    pairs.append((agent_a, agent_b))
                    logger.debug(f"Created game {agent_a} vs {agent_b}")
                    if self.bidirectional:
                        pairs.append((agent_b, agent_a))
                        used_pairs.add((agent_b, agent_a))

                    # Put it back to try with the next agent_a
                    second_elements.append(agent_b)
                    break

                second_elements.append(agent_b)
        logger.debug(f"Created {len(pairs)} games")
        return pairs

    def _load_prompt(self, prompt_name: str):
        prompts_path = f"prompts/answers/{prompt_name}.txt"
        if not os.path.isfile(prompts_path):
            logger.exception(f"Prompts file {prompts_path} not found")
            raise FileNotFoundError
        with open(prompts_path) as f:
            self.prompt = f.read().strip()

    def _read_queries(self, queries_file: str):
        # TODO: Allow to dynamically add queries from answers
        with open(queries_file, "r") as f:
            reader = csv.reader(f)
            for qid, query in reader:
                # Check if the file has a header
                if "qid" in qid:
                    continue
                self.query_to_qid[query] = qid
                self.qid_to_query[qid] = query

    def _read_reasonings(self, reasonigs_file: str):
        for line in csv.DictReader(open(reasonigs_file)):
            qid = line["qid"]
            did = line["docid"]
            self.reasonings[qid][did] = line["response"]

    def _read_answers(self, ans_path: str):
        # TODO: Simplify file formats
        directory = Path(ans_path)
        for f in directory.glob("*.jsonl"):
            agent_name = f.stem
            self.players.add(agent_name)
            for line in f.open():
                data = json.loads(line)
                query = data["query"]
                if "qid" not in data:
                    if query not in self.query_to_qid:
                        logger.debug(
                            f"Query {query} is not in the queries file. Skipping..."
                        )
                        continue
                    qid = self.query_to_qid[query]
                else:
                    qid = data["qid"]
                    self.query_to_qid[query] = qid
                if query not in self.query_to_qid:
                    continue

                self.answers[qid]["Question"] = query
                self.answers[qid][agent_name] = data["response"]["answer"]
        total_players = len(self.players)
        if total_players < 2:
            raise ValueError(
                f"Need at least 2 players to create games. Found {total_players}"
            )
        if (total_players * (total_players - 1)) < self.k:
            possible_games = total_players * (total_players - 1)
            logger.warning(
                f"Requested {self.k} games but only {possible_games} are possible"
            )
            logger.warning(f"Will create {possible_games} games per query instead")
            self.k = possible_games
