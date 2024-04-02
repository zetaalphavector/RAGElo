"""A Retrieval Evaluator is a class that evaluates the results of a retrieval system.
It receives a set of queries used to retrieve a document and their respective retrieved documents,
and returns a score or a label for each document."""

import csv
import dataclasses
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Optional, Type, get_type_hints

from tenacity import RetryError
from tqdm.auto import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types import Document, RetrievalEvaluatorTypes
from ragelo.types.configurations import BaseEvaluatorConfig


class BaseRetrievalEvaluator(BaseEvaluator):
    config: BaseEvaluatorConfig
    output_columns: list[str] = ["qid", "did", "raw_answer", "answer"]
    scoring_key: str = "answer"
    output_file: str = "retrieval_evaluations.csv"

    def __init__(
        self,
        config: BaseEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.config = config
        self.llm_provider = llm_provider
        if config.output_file is not None:
            self.output_file = config.output_file

    def run(
        self, documents: dict[str, dict[str, Document]]
    ) -> dict[str, dict[str, str]]:
        """Evaluate all the documents for each query"""
        use_progress_bar = self.config.verbose
        skip_docs = self.__get_skip_docs()
        answers: dict[str, dict[str, str]] = defaultdict(lambda: dict())
        for qid in tqdm(
            documents.keys(),
            desc="Annotating Documents",
            disable=not use_progress_bar,
            ncols=100,
            leave=False,
            position=0,
        ):
            for document in tqdm(
                documents[qid].values(),
                desc=qid,
                disable=not use_progress_bar,
                ncols=100,
                leave=False,
                position=1,
            ):
                if (qid, document.did) in skip_docs:
                    logger.debug(f"Skipping {qid} {document.did}")
                    continue

                try:
                    answer_dict = self.evaluate_single_sample(document)
                except (RetryError, ValueError):
                    continue
                self._dump_response(answer_dict)

                answers[qid][document.did] = answer_dict["answer"]
        return answers

    def evaluate_single_sample(self, document: Document) -> dict[str, Any]:
        """Evaluates a single query-document pair. Returns the raw answer and the processed answer."""
        message = self._build_message(document)
        try:
            raw_answer = self.llm_provider(message)
        except RetryError as e:
            logger.warning(
                f"Failed to FETCH answers for {document.query.qid} {document.did}"
            )
            raise e
        try:
            answer = self._process_answer(raw_answer)
        except ValueError as e:
            logger.warning(
                f"Failed to PARSE answer for {document.query.qid} {document.did}"
            )
            raise e
        return {
            "qid": document.query.qid,
            "did": document.did,
            "raw_answer": raw_answer,
            "answer": answer,
        }

    def _build_message(self, document: Document) -> str | list[dict[str, str]]:
        """Builds the prompt to send to the LLM."""
        raise NotImplementedError

    @abstractmethod
    def _process_answer(self, answer: str) -> Any:
        """Processes the LLM evaluator output into some serializable format"""
        raise NotImplementedError

    def __get_skip_docs(self) -> set[tuple[str, str]]:
        """Skips documents that have already been annotated"""
        skip_docs = set()
        if self.config.force and os.path.isfile(self.output_file):
            logger.warning(f"Removing existing {self.output_file}!")
            os.remove(self.output_file)

        if os.path.isfile(self.output_file):
            line: dict[str, str]
            for line in csv.DictReader(
                open(self.output_file), fieldnames=self.output_columns
            ):
                skip_docs.add((line["qid"], line["did"]))

        if len(skip_docs) > 0:
            logger.warning(
                f"Skipping {len(skip_docs)} documents already annotated! "
                "If you want to re-annotate them, please use the --force flag"
            )
        return skip_docs

    def _print_response(self, answer_dict: dict[str, str]):
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
                logger.warning("Rich not installed. Using plain print")
                self.config.rich_print = False

        else:
            for key in answer_dict:
                tqdm.write(f"{key.capitalize()}: {answer_dict[key]}")

    def _dump_response(self, answer_dict: dict[str, str], file: Optional[str] = None):
        output_file = file if file else self.output_file
        if not os.path.isfile(output_file):
            logger.debug(f"Creating new file {output_file}")
            with open(output_file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=self.output_columns)
                writer.writeheader()
        self._print_response(answer_dict)
        with open(output_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.output_columns)
            writer.writerow(answer_dict)

    @staticmethod
    def _load_from_csv(file_path: str) -> dict[str, str]:
        """extra content from a CSV file"""
        contents = {}
        for line in csv.reader(open(file_path, "r")):
            contents[line[0]] = line[1]
        return contents

    @classmethod
    def from_config(cls, config: BaseEvaluatorConfig, llm_provider: BaseLLMProvider):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> Type[BaseEvaluatorConfig]:
        return get_type_hints(cls)["config"]


class RetrievalEvaluatorFactory:
    registry: dict[RetrievalEvaluatorTypes | str, Type[BaseRetrievalEvaluator]] = {}

    @classmethod
    def register(cls, evaluator_name: RetrievalEvaluatorTypes) -> Callable:
        def inner_wrapper(
            wrapped_class: Type[BaseRetrievalEvaluator],
        ) -> Type[BaseRetrievalEvaluator]:
            if evaluator_name in cls.registry:
                logger.warning(f"Overwriting {evaluator_name} in registry")
            cls.registry[evaluator_name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        evaluator_name: RetrievalEvaluatorTypes | str,
        llm_provider: BaseLLMProvider | str,
        config: Optional[BaseEvaluatorConfig] = None,
        **kwargs,
    ) -> BaseRetrievalEvaluator:
        if isinstance(llm_provider, str):
            llm_provider_instance = get_llm_provider(llm_provider, **kwargs)
        else:
            llm_provider_instance = llm_provider
        if evaluator_name not in cls.registry:
            raise ValueError(
                f"Unknown retrieval evaluator {evaluator_name}\n"
                f"Valid options are {list(cls.registry.keys())}"
            )
        if config is None:
            class_ = cls.registry[evaluator_name]
            type_config = class_.get_config_class()
            valid_keys = [field.name for field in dataclasses.fields(type_config)]
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            config = type_config(**valid_args)
        return cls.registry[evaluator_name].from_config(config, llm_provider_instance)


def get_retrieval_evaluator(
    evaluator_name: RetrievalEvaluatorTypes | str,
    llm_provider: BaseLLMProvider | str,
    config: Optional[BaseEvaluatorConfig] = None,
    **kwargs,
) -> BaseRetrievalEvaluator:
    return RetrievalEvaluatorFactory.create(
        evaluator_name,
        llm_provider=llm_provider,
        config=config,
        **kwargs,
    )
