import csv
import json
import os
import random
import re
from collections import defaultdict, deque
from typing import Dict, List, Tuple

from tenacity import RetryError

from ragelo.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluator,
    AnswerEvaluatorFactory,
)
from ragelo.logger import logger


@AnswerEvaluatorFactory.register("PairwiseWithReasoning")
class PairwiseWithReasoning(AnswerEvaluator):
    """A evaluator that evaluates RAG-based answers pairwise, with document reasoning"""

    prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants tasked to answer the question displayed below, based on a set of documents retrieved by a search engine.
    You should choose the assistant that best answers the user question based on a set of reference documents that may or not be relevant.
    Answers cite documents using square brackets.  For each reference document, you will be provided with a reasoning explaining why the document is or is not relevant.
    Your evaluation should consider factors such as the correctness, helpfulness, completeness, accuracy, depth, and level of detail of their responses. Details are only useful if they answer the user question. If an answers contains non-relevant details, it should not be preferred over one that only use relevant information.
    Begin your evaluation by explaining why each answer correctly answers the user question. Then, you should compare the two responses and provide a short explanation on their differences. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
    [User Question]
    {query}
    [Reference Documents]
    {documents}
    [The Start of Assistant A's Answer]
    {answer_a}
    [The End of Assistant A's Answer]
    [The Start of Assistant B's Answer]
    {answer_b}
    [The End of Assistant B's Answer]"""  # noqa: E501

    def __init__(
        self,
        reasonings_file: str,
        k: int = 100,
        bidirectional: bool = False,
        *args,
        **kwargs,
    ):
        self.k = k
        super().__init__(*args, **kwargs)
        self.bidirectional = bidirectional
        self.pattern = re.compile(r"\[\[([^]]+)]].*$(?:(?!\[\[).)*", re.DOTALL)
        self.reasonings = self._load_reasonings(reasonings_file)
        self.evaluations: List[Dict[str, str]] = []

    def run(self) -> List[Dict[str, str]]:
        unparsed_answers = 0
        skip_tuples = set()
        self.prompts = self.__create_all_prompts()
        if os.path.exists(self.output_file) and not self.force:
            for line in open(self.output_file):
                data = json.loads(line)
                qid = data["query_id"]
                agent_a = data["agent_a"]
                agent_b = data["agent_b"]
                skip_tuples.add((qid, agent_a, agent_b))
            logger.info(f"Skipping {len(skip_tuples)} games...")
        if self.force and os.path.exists(self.output_file):
            # remove the file
            os.remove(self.output_file)
        if len(self.prompts) - len(skip_tuples) > 0:
            logger.info(f"Running {len(self.prompts) - len(skip_tuples)} games...")

        if self.print:
            try:
                from rich.progress import track

                p_iterator = track(self.prompts, description="Evaluating games")
            except ImportError:
                p_iterator = self.prompts
        for p in p_iterator:
            qid = p["query_id"]
            agent_a = p["agent_a"]
            agent_b = p["agent_b"]
            prompt_str = p["prompt"]
            if (qid, agent_a, agent_b) in skip_tuples:
                continue
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
                logger.info(
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
                logger.info("[bold white] Evaluator answer: [/bold white]")
                logger.info(parser_ans)
            with open(self.output_file, "a") as f:
                d = {
                    "query_id": qid,
                    "agent_a": agent_a,
                    "agent_b": agent_b,
                    "prompt": prompt_str,
                    "answer": gpt_answer,
                    "relevant": relevant,
                }
                json.dump(d, f)
                f.write("\n")
                self.evaluations.append(d)

        logger.info("✅ Done!")
        logger.info(f"Unparsed answers: {unparsed_answers}")
        logger.info(f"Total evaluations: {len(self.prompts) - unparsed_answers}")
        return self.evaluations

    def __generate_random_games(self) -> List[Tuple[str, str]]:
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
        first_agents_q = deque(first_agents)
        second_agents_q = deque(second_agents)

        used_pairs = set()  # avoid re-using pairs
        pairs = []
        while first_agents_q and len(pairs) < self.k:
            agent_a = first_agents_q.popleft()

            for _ in range(len(second_agents)):
                agent_b = second_agents_q.popleft()
                if agent_b != agent_a and (agent_a, agent_b) not in used_pairs:
                    used_pairs.add((agent_a, agent_b))
                    pairs.append((agent_a, agent_b))
                    logger.debug(f"Created game {agent_a} vs {agent_b}")
                    if self.bidirectional:
                        pairs.append((agent_b, agent_a))
                        used_pairs.add((agent_b, agent_a))
                        logger.debug(f"Created game {agent_b} vs {agent_a}")
                    second_agents_q.append(agent_b)
                    break
                second_agents_q.append(agent_b)
        logger.info(f"Created {len(pairs)} games")
        return pairs

    def __create_all_prompts(self) -> List[Dict[str, str]]:
        prompts = []
        random_pairs = self.__generate_random_games()
        if self.print:
            try:
                from rich.progress import track

                a_iterator = track(self.answers, description="Creating prompts")
            except ImportError:
                a_iterator = self.answers
        for qid in a_iterator:
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
                    {"query_id": qid, "agent_a": a, "agent_b": b, "prompt": prompt}
                )
        return prompts

    def _check_validity(self):
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

    def _load_reasonings(self, reasonings_path: str) -> Dict[str, Dict[str, str]]:
        reasonings: Dict[str, Dict[str, str]] = defaultdict(lambda: dict())
        reasonings_read = 0
        for line in csv.DictReader(open(reasonings_path)):
            if line["query_id"] not in self.queries:
                continue
            reasonings_read += 1
            reasonings[line["query_id"]][line["did"]] = line["answer"]
        logger.info(f"Loaded {reasonings_read} reasonings")
        return dict(reasonings)
