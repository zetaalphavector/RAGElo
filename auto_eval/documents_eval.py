"""Manages document evaluations."""
import csv
import os
from typing import Dict, List

from loguru import logger
from rich import print
from rich.progress import track

from auto_eval.opeanai_client import OpenAiClient


class DocumentEvaluator:
    """Evaluates documents with a LLM judge"""

    def __init__(
        self,
        documents_path: str,
        prompt_name: str = "prompt_relevancy",
        model_name: str = "gpt-4",
        credentials_file: str | None = None,
        print_answers: bool = False,
        force: bool = False,
    ):
        self.print_answers = print_answers
        self.force = force

        if credentials_file and os.path.isfile(credentials_file):
            logger.info(f"Loading credentials from {credentials_file}")
            with open(credentials_file) as f:
                for line in f:
                    key, value = line.strip().split("=")
                    logger.debug(f"Setting {key} from file")
                    os.environ[key] = value

        self.openai_client = OpenAiClient(model=model_name)
        self.load_prompt(prompt_name)
        self.rows = {}
        self.load_documents(documents_path)

    def load_prompt(self, prompt_name: str):
        prompts_path = f"prompts/retrieval/{prompt_name}.txt"
        if not os.path.isfile(prompts_path):
            logger.exception(f"Prompts file {prompts_path} not found")
            raise FileNotFoundError

        with open(prompts_path) as f:
            self.prompt = f.read().strip()

    def load_documents(self, documents_path: str):
        if not os.path.isfile(documents_path):
            logger.exception(f"Documents file {documents_path} not found")
            raise FileNotFoundError

        for line in csv.DictReader(open(documents_path)):
            qid = line["query_id"]
            did = line["doc_id"]

            if (qid, did) not in self.rows:
                self.rows[(qid, did)] = {
                    "title": line["title"],
                    "passage": line["passage"],
                    "query": line["Query"],
                }
            else:
                self.rows[(qid, did)]["passage"] += "\n" + line["passage"]

        logger.info(f"Loaded {len(self.rows)} documents")

    def get_answers(self, output_path: str):
        # TODO: Allow for resuming from a checkpoint and using tenacity to retry
        if os.path.isfile(output_path) and not self.force:
            logger.warning(
                "Cowardly refusing to call OpenAI. "
                f"Output file {output_path} already exists. Use --force to overwrite."
            )
            return
        results = []

        for row in track(self.rows, description="Annotating documents"):
            message = self.build_message(self.rows[row])
            resp = self.openai_client(message)
            if self.print_answers:
                print(
                    "[bold cyan]Query       [/bold cyan]: "
                    f"[not bold cyan]{self.rows[row]['query']}[/not bold cyan]"
                )
                print(f"[bold cyan]Document ID [/bold cyan]: {row[1]}")
                print(
                    "[bold cyan]Evaluation  [/bold cyan]: "
                    f"[not bold]{resp}[/not bold]"
                )
                print()

            results.append(
                {
                    "qid": row[0],
                    "query": self.rows[row]["query"],
                    "docid": row[1],
                    "title": self.rows[row]["title"],
                    "passage": self.rows[row]["passage"],
                    "response": resp,
                }
            )

        with open(output_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    def build_message(self, row: Dict[str, str]) -> List[Dict[str, str]]:
        return [
            {
                "role": "user",
                "content": self.prompt.format(
                    user_question=row["query"],
                    doc_title=row["title"],
                    doc_content=row["passage"],
                ),
            },
        ]
