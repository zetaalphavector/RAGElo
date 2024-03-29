import csv
import logging
import os
import random
import re
from collections import defaultdict, deque
from typing import Optional

from tenacity import RetryError
from tqdm.auto import tqdm

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
    BaseAnswerEvaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import AgentAnswer
from ragelo.types.configurations import AnswerEvaluatorConfig


@AnswerEvaluatorFactory.register("pairwise_reasoning")
class PairwiseWithReasoningEvaluator(BaseAnswerEvaluator):
    """A evaluator that evaluates RAG-based answers pairwise, with document reasoning"""

    output_columns: list[str] = ["qid", "agent_a", "agent_b", "raw_answer", "answer"]
    prompt = """
Please act as an impartial judge and evaluate the quality of the responses provided \
by two AI assistants tasked to answer the question displayed below, based on a set \
of documents retrieved by a search engine.
You should choose the assistant that best answers the user question based on a set \
of reference documents that may or not be relevant.
Answers cite documents using square brackets. For each reference document, you will \
be provided with a reasoning explaining why the document is or is not relevant.
Your evaluation should consider factors such as the correctness, helpfulness, \
completeness, accuracy, depth, and level of detail of their responses.\
Details are only useful if they answer the user question. If an answer \
contains non-relevant details, it should not be preferred over one that only \
use relevant information.
Begin your evaluation by explaining why each answer correctly answers the user \
question. Then, you should compare the two responses and provide a short explanation \
on their differences. Avoid any position biases and ensure that the order in which \
the responses were presented does not influence your decision. Do not allow the \
length of the responses to influence your evaluation. Be as objective as possible.
After providing your explanation, output your final verdict by strictly following \
this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, \
and "[[C]]" for a tie.

[User Question]
{query}

[Reference Documents]
{documents}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
""".strip()

    def __init__(
        self,
        config: AnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        if not self.config.output_file:
            self.output_file = "pairwise_reasoning_evaluator.csv"
        else:
            self.output_file = self.config.output_file
        if not self.config.reasoning_file:
            raise ValueError("Reasoning file is required for PairwiseWithReasoning")
        self.k = self.config.k
        self.bidirectional = self.config.bidirectional
        self.pattern = re.compile(r"\[\[([^]]+)]].*$(?:(?!\[\[).)*", re.DOTALL)
        self.reasoning = self.__load_reasonings(self.config.reasoning_file)

    def run(self, answers: dict[str, list[AgentAnswer]]) -> list[dict[str, str]]:
        use_progress_bar = self.config.verbose
        unparsed_answers = 0
        skip_tuples = self.__get_skip_tuples()
        evaluations: list[dict[str, str]] = []
        tuples = self.__prepare_all_tuples(answers, use_progress_bar)
        if len(tuples) - len(skip_tuples) > 0:
            logging.info(f"Running {len(tuples) - len(skip_tuples)} games...")
        for answer_a, answer_b in tqdm(
            tuples,
            desc="Evaluating games",
            disable=not self.config.verbose,
            leave=False,
            position=0,
            ncols=100,
        ):
            qid = answer_a.query.qid
            agent_a = answer_a.agent
            agent_b = answer_b.agent
            if (qid, agent_a, agent_b) in skip_tuples:
                logging.debug(f"Skipping {qid} {agent_a} {agent_b}")
                continue
            try:
                answer_dict = self.evaluate_single_sample((answer_a, answer_a))
            except RetryError:
                continue
            except ValueError:
                unparsed_answers += 1
                continue
            self._dump_response(answer_dict)
            evaluations.append(answer_dict)

        if self.config.verbose:
            print("✅ Done!")
            print(f"Unparsed answers: {unparsed_answers}")
            print(f"Total evaluations: {len(evaluations)}")
        return evaluations

    def evaluate_single_sample(
        self, answer: tuple[AgentAnswer, AgentAnswer]
    ) -> dict[str, str]:
        qid = answer[0].query.qid
        agent_a_id = answer[0].agent
        agent_b_id = answer[1].agent
        reasoning = "\n".join(
            [" ".join([f"[{idx}]", r]) for (idx, r) in self.reasoning[qid].items()]
        )
        answer_agent_a = answer[0].text
        answer_agent_b = answer[1].text

        prompt = self.prompt.format(
            query=answer[0].query.query,
            documents=reasoning,
            answer_a=answer_agent_a,
            answer_b=answer_agent_b,
        )
        try:
            raw_answer = self.llm_provider(prompt)
        except RetryError as e:
            logging.warning(
                f"Failed to FETCH answers for {qid} {agent_a_id}, {agent_b_id}"
            )
            raise e
        try:
            processed_answer = self._process_answer(raw_answer)
        except ValueError as e:
            logging.warning(
                f"Failed extracting answer for {qid}, {agent_a_id}, {agent_b_id}."
                "Probably not enough tokens in the answer."
                f"Full answer:\n{raw_answer}",
            )
            raise e
        return {
            "qid": qid,
            "agent_a": agent_a_id,
            "agent_b": agent_b_id,
            "raw_answer": raw_answer,
            "answer": processed_answer,
        }

    def _dump_response(self, answer_dict: dict[str, str], file: Optional[str] = None):
        output_file = file if file else self.output_file
        if not os.path.isfile(output_file):
            logging.debug(f"Creating new file {output_file}")
            with open(output_file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=self.output_columns)
                writer.writeheader()
        with open(output_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.output_columns)
            writer.writerow(answer_dict)
        self._print_response(answer_dict)

    def _print_response(self, answer_dict: dict[str, str]):
        if not self.config.verbose:
            return
        qid = answer_dict["qid"]
        agent_a = answer_dict["agent_a"]
        agent_b = answer_dict["agent_b"]
        raw_answer = answer_dict["raw_answer"]
        relevant = answer_dict["answer"]
        if self.config.rich_print:
            try:
                import rich

                rich.print(
                    f"[not bold white][{qid}][/not bold white] "
                    f"[bold blue] {agent_a:<18} [/bold blue] 🆚  "
                    f"[bold red] {agent_b}[/bold red]"
                )
                rich.print("[bold white] Evaluator answer: [/bold white]")
                if relevant == "A":
                    parser_ans = raw_answer.replace("[[A]]", "[bold blue]A[/bold blue]")
                elif relevant == "B":
                    parser_ans = raw_answer.replace("[[B]]", "[bold red]B[/bold red]")
                else:
                    parser_ans = raw_answer.replace(
                        "[[C]]", "[bold purple]C[/bold purple]"
                    )
                rich.print(parser_ans)
            except ImportError:
                logging.warning("Rich not installed. Using plain print")
                self.config.rich_print = False
        else:
            tqdm.write(f"{qid}: {agent_a} vs {agent_b}")
            tqdm.write(f"Evaluator full answer: {raw_answer}")
            tqdm.write(f"Relevant answer: {relevant}")

    def __generate_random_games(
        self, answers: dict[str, list[AgentAnswer]]
    ) -> list[tuple[str, str]]:
        """Creates all prompts necessary for running the evaluator"""
        all_agents = list({x.agent for ans in answers.values() for x in ans})
        total_agents = len(all_agents)
        if self.k == -1:
            logging.info("Creating all possible games...")
            pairs = []
            for i in range(total_agents):
                for j in range(i + 1, total_agents):
                    pairs.append((all_agents[i], all_agents[j]))
            random.shuffle(pairs)
            return pairs
        rounds_per_agent = self.k // total_agents
        leftover_rounds = self.k % total_agents

        extra_samples_1 = random.sample(all_agents, k=leftover_rounds)
        extra_samples_2 = random.sample(all_agents, k=leftover_rounds)

        first_agents = all_agents * rounds_per_agent + extra_samples_1
        second_agents = all_agents * rounds_per_agent + extra_samples_2

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
                    logging.debug(f"Created game {agent_a} vs {agent_b}")
                    if self.bidirectional:
                        pairs.append((agent_b, agent_a))
                        used_pairs.add((agent_b, agent_a))
                        logging.debug(f"Created game {agent_b} vs {agent_a}")
                    second_agents_q.append(agent_b)
                    break
                second_agents_q.append(agent_b)
        logging.info(f"Created {len(pairs)} games")
        return pairs

    def __prepare_all_tuples(
        self,
        answers: dict[str, list[AgentAnswer]],
        use_progress_bar: bool = True,
    ) -> list[tuple[AgentAnswer, AgentAnswer]]:
        all_tuples = []
        random_pairs = self.__generate_random_games(answers)

        answers_per_agent: dict[str, dict[str, AgentAnswer]] = defaultdict(dict)
        queries_per_agent: defaultdict[str, set] = defaultdict(set)
        for qid, answers_list in answers.items():
            for answer in answers_list:
                answers_per_agent[answer.agent][qid] = answer
                queries_per_agent[answer.agent].add(qid)

        for qid in tqdm(
            answers,
            desc="Creating prompts",
            disable=not use_progress_bar,
            ncols=100,
            leave=False,
        ):
            for a, b in random_pairs:
                if qid not in queries_per_agent[a] or qid not in queries_per_agent[b]:
                    continue
                all_tuples.append(
                    (answers_per_agent[a][qid], answers_per_agent[b][qid])
                )
        return all_tuples

    def __get_skip_tuples(self) -> set[tuple[str, str, str]]:
        skip_tuples = set()
        if self.config.force and os.path.exists(self.output_file):
            logging.warning(f"Removing existing {self.output_file}!")
            os.remove(self.output_file)

        if os.path.isfile(self.output_file):
            line: dict[str, str]
            for line in csv.DictReader(
                open(self.output_file), fieldnames=self.output_columns
            ):
                skip_tuples.add((line["qid"], line["agent_a"], line["agent_b"]))
        if len(skip_tuples) > 0:
            logging.warning(
                f"Skipping {len(skip_tuples)} games already evaluated! "
                "If you want to re-evaluate them, please use the --force flag"
            )
        return skip_tuples

    def _process_answer(self, answer: str) -> str:
        """Extracts the relevant part of an answer."""
        match_ans = self.pattern.search(answer)
        if not match_ans:
            raise ValueError(f"Could not find answer in {answer}")
        answer = match_ans.group(1)
        if answer not in ["A", "B", "C"]:
            raise ValueError(f"Unknown answer: {answer}")
        return answer

    def __load_reasonings(
        self,
        reasoning_path: str,
        query_id_col: str = "query_id",
        document_id_col: str = "did",
        answer_col: str = "answer",
    ) -> dict[str, dict[str, str]]:
        reasoning: dict[str, dict[str, str]] = defaultdict(lambda: dict())
        reasoning_read = 0
        for line in csv.DictReader(open(reasoning_path)):
            reasoning_read += 1
            reasoning[line[query_id_col]][line[document_id_col]] = line[answer_col]
        logging.info(f"Loaded {reasoning_read} reasonings")
        return dict(reasoning)
