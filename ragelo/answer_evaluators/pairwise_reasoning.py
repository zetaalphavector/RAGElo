import csv
import json
import os
import random
import re
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

from ragelo.answer_evaluators.base_answer_evaluator import (
    AnswerAnnotation,
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
        self.reasonings = self.__load_reasonings(reasonings_file)

    def _build_message(self, sample: AnswerAnnotation) -> str:
        if sample.agent_b is None:
            raise ValueError("Pairwise evaluator needs two agents!")
        query = self.queries[sample.qid]
        reasonings = "\n".join(
            list(
                [
                    " ".join([f"[{a}]", b])
                    for (a, b) in self.reasonings[sample.qid].items()
                ]
            )
        )
        ans_a = self.answers[sample.qid][sample.agent_a].strip()
        ans_b = self.answers[sample.qid][sample.agent_b].strip()
        return self.prompt.format(
            query=query,
            documents=reasonings,
            answer_a=ans_a,
            answer_b=ans_b,
        )

    def _extract_relevant(self, answer: str) -> str:
        match_ans = self.pattern.search(answer)
        if not match_ans:
            logger.warning(
                "Failed extracting relevant from answer!"
                "Probably not enough tokens in the answer."
                f"Full answer:\n{answer}",
            )
            raise ValueError(f"Could not find answer in {answer}")

        answer = match_ans.group(1)
        if answer not in ["A", "B", "C"]:
            raise ValueError(f"Unknown answer: {answer}")
        return answer

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

    def _get_skip_tuples(self) -> Set[AnswerAnnotation]:
        skip_tuples = set()
        if os.path.isfile(self.output_file) and not self.force:
            for line in open(self.output_file):
                data = json.loads(line)
                skip_tuples.add(
                    AnswerAnnotation(
                        qid=data["query_id"],
                        agent_a=data["agent_a"],
                        agent_b=data["agent_b"],
                    )
                )
        if len(skip_tuples) > 0:
            logger.warning(
                f"Skipping {len(skip_tuples)} tuples already annotated! "
                "If you want to reannotate them, please use the --force flag"
            )
        return skip_tuples

    def _sample_answers(self) -> List[AnswerAnnotation]:
        agent_pairs = []
        random_pairs = self.__generate_random_games()
        for qid in self.answers:
            for a, b in random_pairs:
                if a not in self.answers[qid] or b not in self.answers[qid]:
                    if a not in self.answers[qid]:
                        logger.warning(f"Missing answers from agent {a} to query {qid}")
                    elif b not in self.answers[qid]:
                        logger.warning(f"Missing answers from agent {b} to query {qid}")
                    continue
                agent_pairs.append(AnswerAnnotation(qid=qid, agent_a=a, agent_b=b))
        return agent_pairs

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

    def __load_reasonings(self, reasonings_path: str) -> Dict[str, Dict[str, str]]:
        """Loads the reasonigs of the relevace of each document for each query"""
        reasonings: Dict[str, Dict[str, str]] = defaultdict(lambda: dict())
        reasonings_read = 0
        for line in csv.DictReader(open(reasonings_path)):
            if line["query_id"] not in self.queries:
                continue
            reasonings_read += 1
            reasonings[line["query_id"]][line["did"]] = line["answer"]
        logger.info(f"Loaded {reasonings_read} reasonings")
        return dict(reasonings)
