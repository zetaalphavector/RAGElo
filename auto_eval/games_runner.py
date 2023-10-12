import csv
import json
import os
import re

from loguru import logger
from rich import print
from rich.progress import track
from tenacity import RetryError

from auto_eval.opeanai_client import OpenAiClient


class GamesRunner:
    def __init__(
        self,
        games_file: str,
        output_file: str,
        answers_file: str,
        model_name: str = "gpt-4",
        credentials_file: str | None = None,
        print_answers: bool = False,
        force: bool = False,
    ):
        """Initializes a GameRunner for calling an LLM-based evaluator on a list of games.
        Args:
            games_file: A jsonl file with a list of games, one per line.
            output_file: The csv file to write the results to.
            answers_file: jsonl file to write the answers to.
            model_name: The name of the model to use. Defaults to "gpt-4".
            credentials_file: A file with OpenAI credentials. Defaults to None.
            print_answers: Whether to print the answers to the screen. Defaults to False.
            force: Whether to overwrite the output file if it exists. Defaults to False.


        """
        self.force = force
        self.print_answers = print_answers
        self.output_file = output_file
        self.answers_file = answers_file
        self.pattern = re.compile(r"\[\[([^]]+)]].*$(?:(?!\[\[).)*", re.DOTALL)
        self.score_map = {"A": 1, "B": 0, "C": 0.5}

        if credentials_file and os.path.isfile(credentials_file):
            logger.info(f"Loading credentials from {credentials_file}")
            with open(credentials_file) as f:
                for line in f:
                    key, value = line.strip().split("=")
                    logger.debug(f"Setting {key} from file")
                    os.environ[key] = value

        self.openai_client = OpenAiClient(model=model_name)

        self.prompts = []
        for line in open(games_file):
            self.prompts.append(json.loads(line))
        total_prompts = len(self.prompts)
        logger.info(f"Loaded {total_prompts} games")

    def run_games(self):
        """Runs the games and writes the results to the output file."""
        unparsed_answers = 0
        skip_tuples = set()
        if os.path.exists(self.answers_file) and not self.force:
            for line in open(self.answers_file):
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
                relevant = self.extract_relevant(gpt_answer)
            except ValueError:
                unparsed_answers += 1
                logger.warning(
                    f"Failed extracting answer for {qid}, {agent_a}, {agent_b}."
                    "Probably not enough tokens in the answer."
                    f"Full answer:\n{gpt_answer}",
                )
                continue
            if self.print_answers:
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
            with open(self.answers_file, "a") as f:
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
            if not os.path.exists(self.output_file):
                with open(self.output_file, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["agent_a", "agent_b", "score"])
            with open(self.output_file, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        agent_a,
                        agent_b,
                        self.score_map[relevant],
                    ]
                )
        print(f":check_mark: Done!")
        print(f"Unparsed answers: {unparsed_answers}")
        print(f"Total evaluations: {len(self.prompts) - unparsed_answers}")

    def extract_relevant(self, answer: str) -> str:
        """Extracts the relevant part of an answer."""
        match_ans = self.pattern.search(answer)
        if not match_ans:
            raise ValueError(f"Could not find answer in {answer}")
        answer = match_ans.group(1)
        if answer not in ["A", "B", "C"]:
            raise ValueError(f"Unknown answer: {answer}")
        return answer
