"""A Retrieval Evaluator is a class that evaluates the results of a retrieval system.
It receives a set of queries used to retrieve a document and their respective retrieved documents,
and returns a score or a label for each document."""

import csv
import dataclasses
import os
from abc import abstractmethod
from typing import Any, Callable, Optional, Type, get_type_hints

from tenacity import RetryError
from tqdm.auto import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types import Document, EvaluatorAnswer, Query, RetrievalEvaluatorTypes
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

    def batch_evaluate(self, queries: list[Query]) -> list[EvaluatorAnswer]:
        """Evaluate all the documents for a list of queries"""
        use_progress_bar = self.config.verbose
        skip_docs = self.__get_skip_docs()
        answers: list[EvaluatorAnswer] = []
        for query in tqdm(
            queries,
            desc="Evaluating retrieved documents",
            disable=not use_progress_bar,
            ncols=100,
            leave=False,
            position=0,
        ):
            for document in tqdm(
                query.retrieved_docs,
                desc=query.qid,
                ncols=100,
                leave=False,
                position=1,
            ):
                qid = query.qid
                did = document.did
                if (qid, did) in skip_docs:
                    logger.debug(f"Skipping {qid} {did}")
                    continue

                try:
                    raw_answer, answer = self.evaluate(query, document)
                except (RetryError, ValueError):
                    continue

                answers.append(
                    EvaluatorAnswer(
                        qid=qid,
                        did=did,
                        raw_answer=raw_answer,
                        answer=answer,
                    )
                )
            self._dump_response(answers[-1])
        return answers

    def evaluate(
        self,
        query: Query | str,
        document: Document | str,
        query_metadata: Optional[dict[str, Any]] = None,
        doc_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Any]:
        """Evaluates a single query-document pair. Returns the raw answer and the processed answer."""
        if isinstance(query, str):
            query = Query(qid="<no_qid>", query=query, metadata=query_metadata)
        if isinstance(document, str):
            document = Document(did="<no_did>", text=document, metadata=doc_metadata)

        message = self._build_message(query, document)
        try:
            raw_answer = self.llm_provider(message)
        except RetryError as e:
            logger.warning(
                f"Failed to FETCH answers for qid: {query.qid} did: {document.did}"
            )
            raise e
        try:
            answer = self._process_answer(raw_answer)
        except ValueError as e:
            logger.warning(
                f"Failed to PARSE answer for qid: {query.qid} did: {document.did}"
            )
            raise e
        return raw_answer, answer

    def _build_message(
        self, query: Query, document: Document
    ) -> str | list[dict[str, str]]:
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

    def _print_response(self, answer_dict: EvaluatorAnswer):
        if not self.config.verbose:
            return
        if self.config.rich_print:
            try:
                import rich

                rich.print(f"[bold blue]🔎 Query ID[/bold blue]: {answer_dict.qid}")
                rich.print(f"[bold blue]📜 Document ID[/bold blue]: {answer_dict.did}")
                rich.print(
                    f"[bold blue]Raw Answer[/bold blue]: {answer_dict.raw_answer}"
                )
                rich.print(
                    f"[bold blue]Parsed Answer[/bold blue]: {answer_dict.answer}"
                )
                rich.print("")

            except ImportError:
                logger.warning("Rich not installed. Using plain print")
                self.config.rich_print = False

        else:
            tqdm.write(f"Query ID: {answer_dict.qid}")
            tqdm.write(f"Document ID: {answer_dict.did}")
            tqdm.write(f"Raw Answer: {answer_dict.raw_answer}")
            tqdm.write(f"Parsed Answer: {answer_dict.answer}")
            tqdm.write("")

    def _dump_response(self, eval_answer: EvaluatorAnswer, file: Optional[str] = None):
        self._print_response(eval_answer)
        if not self.config.write_output:
            return
        output_file = file if file else self.output_file
        if not os.path.isfile(output_file):
            logger.debug(f"Creating new file {output_file}")
            with open(output_file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=self.output_columns)
                writer.writeheader()
        with open(output_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.output_columns)
            writer.writerow(dataclasses.asdict(eval_answer))

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
                logger.debug(f"Overwriting {evaluator_name} in registry")
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
