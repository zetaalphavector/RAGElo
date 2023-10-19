"""Base model for dealing with document evaluators"""
import csv
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Type

from tenacity import RetryError

from ragelo.logger import logger
from ragelo.opeanai_client import OpenAiClient, set_credentials_from_file


class DocumentEvaluator:
    def __init__(
        self,
        query_path: str,
        documents_path: str,
        output_file: str,
        model_name: str = "gpt-4",
        credentials_file: str | None = None,
        verbose: bool = False,
        force: bool = False,
        *args,
        **kwargs,
    ):
        self.verbose = verbose
        self.force = force
        self.output_file = output_file
        self.queries = self._load_queries(query_path)
        self.documents = self._load_documents(documents_path)

        if credentials_file:
            set_credentials_from_file(credentials_file)

        self.openai_client = OpenAiClient(model=model_name)

    def get_answers(self):
        """Runs the evaluator and saves the results to a file"""
        skip_docs = set()
        if os.path.isfile(self.output_file) and not self.force:
            for line in csv.reader(open(self.output_file)):
                qid, did, answer = line
                skip_docs.add((qid, did))
        if self.force and os.path.isfile(self.output_file):
            logger.warning(f"Removing existing {self.output_file}!")
            os.remove(self.output_file)
        if len(skip_docs) > 0:
            logger.warning(
                f"Skipping {len(skip_docs)} documents already annotated! "
                "If you want to reannotate them, please use the --force flag"
            )
        q_iterator = self.queries
        if self.verbose:
            try:
                from rich.progress import track

                q_iterator = track(self.queries, description="Annotating Documents")
            except ImportError:
                pass
        for qid in q_iterator:
            for did in self.documents[qid]:
                if (qid, did) in skip_docs:
                    logger.debug(f"Skipping {qid} {did}")
                    continue
                message = self._build_message(qid, did)
                try:
                    answer = self.openai_client(message)
                    answer = self._process_answer(answer)
                except RetryError:
                    logger.warning(f"Failed to fetch answers for document {qid} {did}")
                    continue
                except ValueError:
                    logger.warning(f"Failed to parse answer for document {qid} {did}")
                    continue
                if self.verbose:
                    logger.info(
                        "[bold cyan]Query       [/bold cyan]: "
                        f"[not bold cyan]{self.queries[qid]}[/not bold cyan]"
                    )
                    logger.info(f"[bold cyan]Document ID [/bold cyan]: {did}")
                    logger.info(
                        "[bold cyan]Evaluation  [/bold cyan]: "
                        f"[not bold]{answer}[/not bold]"
                    )
                    logger.info("")
                if not os.path.isfile(self.output_file):
                    with open(self.output_file, "w") as f:
                        writer = csv.writer(f)
                        writer.writerow(["query_id", "did", "answer"])

                with open(self.output_file, "a") as f:
                    writer = csv.writer(f)
                    writer.writerow([qid, did, answer])

    @abstractmethod
    def _build_message(self, qid: str, did: str) -> str:
        """Builds the prompt to send to the LLM."""
        raise NotImplementedError

    @abstractmethod
    def _process_answer(self, answer: str) -> Any:
        """Processes the LLM evaluator output into some serializable format"""
        raise NotImplementedError

    def _load_queries(self, queries_path: str) -> Dict[str, str]:
        queries = {}
        if not os.path.isfile(queries_path):
            logger.exception(f"Queries file {queries_path} not found")
            raise FileNotFoundError

        for line in csv.reader(open(queries_path)):
            if "query" in line:
                continue
            qid, query = line
            queries[qid] = query
        logger.info(f"Loaded {len(queries)} queries")

        return queries

    def _load_documents(self, documents_path: str) -> Dict[str, Dict[str, str]]:
        rows: Dict[str, Dict[str, str]] = defaultdict(lambda: dict())
        if not os.path.isfile(documents_path):
            logger.exception(f"Documents file {documents_path} not found")
            raise FileNotFoundError

        for line in csv.DictReader(open(documents_path)):
            qid = line["query_id"]
            did = line["doc_id"]
            if qid not in self.queries:
                continue
            if did not in rows[qid]:
                rows[qid][did] = line["document_text"]
            else:
                logger.debug(
                    f"Document {did} found again. trying to merge them by title"
                )
                overlap_size = self.__title_overlap(
                    line["document_text"], rows[qid][did]
                )
                if overlap_size > 10:
                    title = line["document_text"][:overlap_size]
                    logger.debug(
                        f"Title overlap found: {title}. Merging documents on title"
                    )
                    new_doc = line["document_text"][overlap_size:]
                    rows[qid][did] += "\n" + new_doc
                else:
                    logger.warning(
                        f"Documents do not share title. Skipping document {did}"
                    )

        logger.info(f"Loaded {len(rows)} documents")
        return rows

    def __load_from_csv(self, file_path: str) -> Dict[str, str]:
        """extra content from a CSV file"""
        contents = {}
        for line in csv.reader(open(file_path, "r")):
            contents[line[0]] = line[1]
        return contents

    def __title_overlap(self, new_doc: str, current_doc: str) -> int:
        size = 0
        min_len = min(len(new_doc), len(current_doc))
        while size < min_len and new_doc[size] == current_doc[size]:
            size += 1
        return size


class DocumentEvaluatorFactory:
    registry: Dict[str, Type[DocumentEvaluator]] = {}

    @classmethod
    def register(cls, evaluator_name: str) -> Callable:
        def inner_wrapper(
            wrapped_class: Type[DocumentEvaluator],
        ) -> Type[DocumentEvaluator]:
            cls.registry[evaluator_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, evaluator_name: str, **kwargs) -> DocumentEvaluator:
        if evaluator_name not in cls.registry:
            raise ValueError(f"Unknown evaluator {evaluator_name}")
        return cls.registry[evaluator_name](prompt_name=evaluator_name, **kwargs)
