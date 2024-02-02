"""Base model for dealing with document evaluators"""

import csv
import logging
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from tenacity import RetryError
from tqdm.auto import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import LLMProvider
from ragelo.types import Document, Query
from ragelo.types.configurations import RetrievalEvaluatorConfig


class RetrievalEvaluator(BaseEvaluator):
    def __init__(
        self,
        config: RetrievalEvaluatorConfig,
        llm_provider: LLMProvider,
        queries: Optional[Dict[str, Query]] = None,
        documents: Optional[Dict[str, Dict[str, Document]]] = None,
    ):
        self.queries = self.load_queries(config.query_path) if not queries else queries
        self.documents = (
            self.load_documents(config.documents_path, self.queries)
            if not documents
            else documents
        )
        if not self.config.output_file:
            self.output_file = f"retrieval_evaluator.log"
        else:
            self.output_file = self.config.output_file

        self.llm_provider = llm_provider
        self.config = config
        self.force = self.config.force
        self.rich_print = self.config.rich_print
        self.verbose = self.config.verbose

    def run(self) -> Dict[str, Dict[str, str]]:
        """Evaluate all the documents for each query"""
        use_progress_bar = self.config.verbose
        skip_docs = self.__get_skip_docs()
        answers: Dict[str, Dict[str, str]] = defaultdict(lambda: dict())
        for qid in tqdm(
            self.queries,
            desc="Annotating Documents",
            disable=not use_progress_bar,
            ncols=100,
        ):
            for did in tqdm(
                self.documents[qid],
                desc=qid,
                disable=not use_progress_bar,
                ncols=100,
                leave=False,
            ):
                if (qid, did) in skip_docs:
                    logging.debug(f"Skipping {qid} {did}")
                    continue

                try:
                    answer = self.evaluate_single_sample(qid, did)
                except (RetryError, ValueError):
                    continue
                self._print_response(qid, did, answer)
                self._dump_response(qid, did, answer)
                answers[qid][did] = answer
        return answers

    def evaluate_single_sample(self, qid: str, did: str) -> str:
        """Evaluates a single query-document pair"""
        message = self._build_message(qid, did)
        try:
            answer = self.llm_provider.query(message)
        except RetryError as e:
            logging.warning(f"Failed to FETCH answers for  {qid} {did}")
            raise e
        try:
            answer = self._process_answer(answer)
        except ValueError as e:
            logging.warning(f"Failed to PARSE answer for {qid} {did}")
            raise e
        return answer

    @abstractmethod
    def _build_message(self, qid: str, did: str) -> str:
        """Builds the prompt to send to the LLM."""
        raise NotImplementedError

    @abstractmethod
    def _process_answer(self, answer: str) -> Any:
        """Processes the LLM evaluator output into some serializable format"""
        raise NotImplementedError

    def __get_skip_docs(self) -> Set[Tuple[str, str]]:
        """Skips documents that have already been annotated"""
        skip_docs = set()
        if os.path.isfile(self.output_file) and not self.force:
            for line in csv.reader(open(self.output_file)):
                qid, did, _ = line
                skip_docs.add((qid, did))
        if self.force and os.path.isfile(self.output_file):
            logging.warning(f"Removing existing {self.output_file}!")
            os.remove(self.output_file)
        if len(skip_docs) > 0:
            logging.warning(
                f"Skipping {len(skip_docs)} documents already annotated! "
                "If you want to re-annotate them, please use the --force flag"
            )
        return skip_docs

    def _print_response(self, qid: str, did: str, answer: str) -> None:
        if not self.verbose:
            return
        if self.rich_print:
            try:
                from rich import print
            except ImportError:
                logging.warning("Rich not installed. Using plain print")
                self.rich_print = False
            print(
                "[bold cyan]Query       [/bold cyan]: [not bold cyan]"
                f"{self.queries[qid]}[/not bold cyan]"
            )
            print(f"[bold cyan]Document ID [/bold cyan]: {did}")
            print(f"[bold cyan]Evaluation  [/bold cyan]: [not bold]{answer}[/not bold]")
            print("")
        else:
            print(
                f"Query: {self.queries[qid]}, Document ID: {did}, Evaluation: {answer}"
            )

    def _dump_response(
        self,
        qid: str,
        did: str,
        answer: str,
        file: str | None = None,
    ) -> None:
        output_file = file if file else self.output_file
        if not os.path.isfile(output_file):
            logging.debug(f"Creating new file {output_file}")
            with open(output_file, "w") as f:
                writer = csv.writer(f)
                writer.writerow(["query_id", "did", "answer"])

        with open(output_file, "a") as f:
            writer = csv.writer(f)
            if isinstance(answer, List):
                answer = "\n".join(answer)
            writer.writerow([qid, did, answer])

    @staticmethod
    def _load_from_csv(file_path: str) -> Dict[str, str]:
        """extra content from a CSV file"""
        contents = {}
        for line in csv.reader(open(file_path, "r")):
            contents[line[0]] = line[1]
        return contents


class RetrievalEvaluatorFactory:
    registry: Dict[str, Type[RetrievalEvaluator]] = {}

    @classmethod
    def register(cls, evaluator_name: str) -> Callable:
        def inner_wrapper(
            wrapped_class: Type[RetrievalEvaluator],
        ) -> Type[RetrievalEvaluator]:
            cls.registry[evaluator_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        evaluator_name: str,
        config: RetrievalEvaluatorConfig,
        llm_provider: LLMProvider,
        queries: Optional[Dict[str, Query]] = None,
        documents: Optional[Dict[str, List[Document]]] = None,
    ) -> RetrievalEvaluator:
        if evaluator_name not in cls.registry:
            raise ValueError(f"Unknown evaluator {evaluator_name}")
        return cls.registry[evaluator_name](config, llm_provider, queries, documents)
