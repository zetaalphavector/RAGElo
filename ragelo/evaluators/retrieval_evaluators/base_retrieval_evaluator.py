"""A Retrieval Evaluator is a class that evaluates the results of a retrieval system.
It receives a set of queries used to retrieve a document and their respective retrieved documents,
and returns a score or a label for each document."""

import csv
import json
import logging
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Set, Tuple, Type

from tenacity import RetryError
from tqdm.auto import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import Document, Query
from ragelo.types.configurations import BaseEvaluatorConfig


class BaseRetrievalEvaluator(BaseEvaluator):
    config: BaseEvaluatorConfig
    queries: Dict[str, Query]
    documents: Dict[str, Dict[str, Document]]
    llm_provider: BaseLLMProvider
    output_file: str
    output_columns: List[str] = ["qid", "did", "raw_answer", "answer"]
    scoring_key: str = "answer"

    def __init__(
        self,
        config: BaseEvaluatorConfig,
        queries: Dict[str, Query],
        documents: Dict[str, Dict[str, Document]],
        llm_provider: BaseLLMProvider,
    ):
        if not queries:
            raise ValueError(
                "You are trying to use a Retrieval Evaluator without providing queries"
            )
        if not documents:
            raise ValueError(
                "You are trying to use a Retrieval Evaluator without providing documents"
            )
        self.config = config
        self.queries = queries
        self.documents = documents
        self.llm_provider = llm_provider
        self.output_file = (
            config.output_file if config.output_file else "retrieval_evaluator.log"
        )

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
            leave=False,
            position=0,
        ):
            for did in tqdm(
                self.documents[qid],
                desc=qid,
                disable=not use_progress_bar,
                ncols=100,
                leave=False,
                position=1,
            ):
                if (qid, did) in skip_docs:
                    logging.debug(f"Skipping {qid} {did}")
                    continue

                try:
                    answer_dict = self.evaluate_single_sample(qid, did)
                except (RetryError, ValueError):
                    continue
                self._dump_response(answer_dict)

                answers[qid][did] = answer_dict["answer"]
        return answers

    def evaluate_single_sample(self, qid: str, did: str) -> Dict[str, Any]:
        """Evaluates a single query-document pair. Returns the raw answer and the processed answer."""
        message = self._build_message(qid, did)
        try:
            raw_answer = self.llm_provider(message)
        except RetryError as e:
            logging.warning(f"Failed to FETCH answers for {qid} {did}")
            raise e
        try:
            answer = self._process_answer(raw_answer)
        except ValueError as e:
            logging.warning(f"Failed to PARSE answer for {qid} {did}")
            raise e
        return {
            "qid": qid,
            "did": did,
            "raw_answer": raw_answer,
            "answer": answer,
        }

    @abstractmethod
    def _build_message(self, qid: str, did: str) -> str | List[Dict[str, str]]:
        """Builds the prompt to send to the LLM."""
        raise NotImplementedError

    @abstractmethod
    def _process_answer(self, answer: str) -> Any:
        """Processes the LLM evaluator output into some serializable format"""
        raise NotImplementedError

    def __get_skip_docs(self) -> Set[Tuple[str, str]]:
        """Skips documents that have already been annotated"""
        skip_docs = set()
        if os.path.isfile(self.output_file) and not self.config.force:
            line: Dict[str, str]
            for line in csv.DictReader(
                open(self.output_file), fieldnames=self.output_columns
            ):
                skip_docs.add((line["qid"], line["did"]))
        if self.config.force and os.path.isfile(self.output_file):
            logging.warning(f"Removing existing {self.output_file}!")
            os.remove(self.output_file)
        if len(skip_docs) > 0:
            logging.warning(
                f"Skipping {len(skip_docs)} documents already annotated! "
                "If you want to re-annotate them, please use the --force flag"
            )
        return skip_docs

    def _print_response(self, answer_dict: Dict[str, str]) -> None:
        if not self.config.verbose:
            return
        if self.config.rich_print:
            try:
                import rich

                for key in answer_dict:
                    if "qid" in key or "query" in key:
                        rich.print(
                            f"[bold magenta]🔎{key.capitalize()}[/bold magenta]: ",
                            f"[not bold magenta]{answer_dict[key]}[/not bold magenta]",
                        )
                    rich.print(
                        f"[bold cyan]{key.capitalize()}[/bold cyan]: "
                        f"[not bold cyan]{answer_dict[key]}[/not bold cyan]"
                    )
                rich.print("")

            except ImportError:
                logging.warning("Rich not installed. Using plain print")
                self.config.rich_print = False

        else:
            for key in answer_dict:
                tqdm.write(f"{key.capitalize()}: {answer_dict[key]}")

    def _dump_response(
        self,
        answer_dict: Dict[str, str],
        file: str | None = None,
    ) -> None:
        output_file = file if file else self.output_file
        if not os.path.isfile(output_file):
            logging.debug(f"Creating new file {output_file}")
            with open(output_file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=self.output_columns)
                writer.writeheader()
        self._print_response(answer_dict)
        with open(output_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.output_columns)
            writer.writerow(answer_dict)

    @staticmethod
    def _load_from_csv(file_path: str) -> Dict[str, str]:
        """extra content from a CSV file"""
        contents = {}
        for line in csv.reader(open(file_path, "r")):
            contents[line[0]] = line[1]
        return contents

    @classmethod
    def from_config(cls, config: BaseEvaluatorConfig, llm_provider):
        queries = cls._load_queries(config.query_path)
        documents = cls.load_documents(config.documents_path, queries)
        return cls(config, queries, documents, llm_provider)

    @staticmethod
    def json_answer_parser(answer: str, key: str) -> Any:
        """Parses a Json answer from the LLM and returns a specific key"""

        # Finds all valid JSON objects in the answer that contain the key
        json_objects = []
        for line in answer.strip().split("\n"):
            try:
                json_object = json.loads(line)
                if key in json_object:
                    json_objects.append(json_object)
            except json.JSONDecodeError:
                pass

        # Assumes the valid JSON object is the last one
        if not json_objects:
            raise ValueError(
                "Answer does not contain a valid json object\n"
                f"with the key {key}\n{answer}"
            )
        json_dict = json_objects[-1]
        return json_dict[key]

    def __len__(self) -> int:
        return len(self.queries)


class RetrievalEvaluatorFactory:
    registry: Dict[str, Type[BaseRetrievalEvaluator]] = {}

    @classmethod
    def register(cls, evaluator_name: str) -> Callable:
        def inner_wrapper(
            wrapped_class: Type[BaseRetrievalEvaluator],
        ) -> Type[BaseRetrievalEvaluator]:
            cls.registry[evaluator_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        evaluator_name: str,
        config: BaseEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ) -> BaseRetrievalEvaluator:
        if evaluator_name not in cls.registry:
            raise ValueError(f"Unknown evaluator {evaluator_name}")
        return cls.registry[evaluator_name].from_config(config, llm_provider)
