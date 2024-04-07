import csv
import dataclasses
import json
import os
import string
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from tqdm.auto import tqdm

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
from ragelo.types import AnswerEvaluatorResult, RetrievalEvaluatorResult
from ragelo.types.configurations import BaseEvaluatorConfig


class BaseEvaluator(ABC):
    config: BaseEvaluatorConfig
    output_file: str

    @abstractmethod
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        config: BaseEvaluatorConfig,
    ):
        raise NotImplementedError

    @staticmethod
    def json_answer_parser(answer: str, key: str) -> str:
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

    @staticmethod
    def json_answer_parser_multifields(answer: str, keys: list[str]) -> dict[str, str]:
        """Parses a Json answer from the LLM and returns the values from multiple fields"""

        # Finds all valid JSON objects in the answer that contain the key
        values = {}
        for line in answer.strip().split("\n"):
            try:
                json_object = json.loads(line)
                for k in keys:
                    if k in json_object:
                        values[k] = json_object[k]
            except json.JSONDecodeError:
                pass

        if len(values) != len(keys):
            raise ValueError(
                "Answer does not contain all necessary keys\n"
                f"Expected {keys}, found {values.keys()}.\n"
                f"Full Answer:\n{answer}"
            )
        # Assumes the valid JSON object is the last one
        return values

    @staticmethod
    def _get_fields_from_string(s: str) -> list[str]:
        """Parse a formatted string and return all the fields in it"""
        field_names = [v[1] for v in string.Formatter().parse(s) if v[1] is not None]
        return field_names

    @staticmethod
    def _get_usable_fields_from_metadata(
        prompt: str, metadata: Optional[dict[str, str]], skip_fields: list[str] = []
    ) -> dict[str, str]:
        """Get the fields from the prompt that are in the metadata"""
        expected_fields = BaseEvaluator._get_fields_from_string(prompt)
        valid_fields = {}
        if metadata is None:
            return valid_fields
        for field in expected_fields:
            if field in metadata and field not in skip_fields:
                valid_fields[field] = metadata[field]
        return valid_fields

    @staticmethod
    def _get_skip_tuples(
        output_file: str,
        tuple_columns: list[str],
        force: bool = False,
    ) -> set[Sequence[str]]:
        skip_tuples: set[Sequence[str]] = set()
        if force and os.path.exists(output_file):
            logger.warning(f"Removing existing {output_file}!")
            os.remove(output_file)
        if not os.path.isfile(output_file):
            return skip_tuples
        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            for line in reader:
                skip_tuples.add(tuple(line[col] for col in tuple_columns))
        if len(skip_tuples) > 0:
            logger.info(
                f"Skipping {len(skip_tuples)} rows in {output_file} "
                "If you want to re-evaluate them, please use the --force flag"
            )
        return skip_tuples

    @staticmethod
    def _print_response(
        response: AnswerEvaluatorResult | RetrievalEvaluatorResult,
        rich_print: bool = False,
    ):
        if rich_print:
            try:
                import rich

                rich.print(f"[bold blue]🔎 Query ID[/bold blue]: {response.qid}")
                if isinstance(response, RetrievalEvaluatorResult):
                    rich.print(f"[bold blue]📜 Document ID[/bold blue]: {response.did}")
                else:
                    rich.print(f"[bold blue]🕵️ Agent[/bold blue]: {response.agent}")
                rich.print(f"[bold blue]Raw Answer[/bold blue]: {response.raw_answer}")
                rich.print(f"[bold blue]Parsed Answer[/bold blue]: {response.answer}")
                rich.print("")
                return
            except ImportError:
                logger.warning("Rich not installed. Using plain print")
        tqdm.write(f"Query ID: {response.qid}")
        if isinstance(response, RetrievalEvaluatorResult):
            tqdm.write(f"Document ID: {response.did}")
        else:
            tqdm.write(f"Agent: {response.agent}")
        tqdm.write(f"Raw Answer: {response.raw_answer}")
        tqdm.write(f"Parsed Answer: {response.answer}")
        tqdm.write("")

    def _dump_response(
        self,
        response: AnswerEvaluatorResult | RetrievalEvaluatorResult,
        output_columns: list[str],
        file: Optional[str] = None,
    ):
        if self.config.verbose:
            self._print_response(response, self.config.rich_print)
        if not self.config.write_output:
            return
        output_file = file if file else self.output_file
        if not os.path.isfile(output_file):
            logger.debug(f"Creating new file {output_file}")
            with open(output_file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=output_columns)
                writer.writeheader()
        with open(output_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=output_columns)
            writer.writerow(dataclasses.asdict(response))
