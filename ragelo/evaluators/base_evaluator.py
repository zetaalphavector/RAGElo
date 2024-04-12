import csv
import json
import os
import string
from abc import ABC, abstractmethod
from typing import Any, Optional

from tqdm.auto import tqdm

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
from ragelo.types import (
    AgentAnswer,
    AnswerEvaluatorResult,
    Document,
    Query,
    RetrievalEvaluatorResult,
)
from ragelo.types.configurations import AnswerFormat, BaseEvaluatorConfig


class BaseEvaluator(ABC):
    config: BaseEvaluatorConfig
    output_file: str
    output_columns: list[str]
    scoring_key: str | list[str]

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
        parsed_answer = {}
        for line in answer.strip().split("\n"):
            try:
                json_object = json.loads(line)
                for k in keys:
                    if k in json_object:
                        parsed_answer[k] = json_object[k]
            except json.JSONDecodeError:
                pass

        if len(parsed_answer) != len(keys):
            raise ValueError(
                "Answer does not contain all necessary keys\n"
                f"Expected {keys}, found {parsed_answer.keys()}.\n"
                f"Full Answer:\n{answer}"
            )
        # Assumes the valid JSON object is the last one
        return parsed_answer

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
        valid_fields: dict[str, str] = {}
        if metadata is None:
            return valid_fields
        for field in expected_fields:
            if field in metadata and field not in skip_fields:
                valid_fields[field] = metadata[field]
        return valid_fields

    @staticmethod
    def __get_existing_output_from_json(output_file: str) -> list[dict[str, str]]:
        return json.load(open(output_file, "r"))

    @staticmethod
    def __get_existing_output_from_jsonl(output_file: str) -> list[dict[str, str]]:
        with open(output_file, "r") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def __get_existing_output_from_csv(output_file: str) -> list[dict[str, str]]:
        existing_lines: list[dict[str, str]] = []
        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            for line in reader:
                existing_lines.append(line)
        return existing_lines

    def _get_existing_output(self) -> list[dict[str, str]]:
        existing_lines: list[dict[str, str]] = []
        if self.config.force and os.path.exists(self.output_file):
            logger.warning(f"Removing existing {self.output_file}!")
            os.remove(self.output_file)
        if not os.path.isfile(self.output_file):
            return existing_lines
        if self.output_file.endswith(".json"):
            return self.__get_existing_output_from_json(self.output_file)
        if self.output_file.endswith(".jsonl"):
            return self.__get_existing_output_from_jsonl(self.output_file)
        else:
            return self.__get_existing_output_from_csv(self.output_file)

    @staticmethod
    def _print_response(
        response: AnswerEvaluatorResult | RetrievalEvaluatorResult,
        rich_print: bool = False,
    ):
        answer: str | dict[str, str] | int
        if isinstance(response.answer, dict):
            # Print the answer in a more readable format
            answer = json.dumps(response.answer, indent=4)
        else:
            answer = response.answer
        if rich_print:
            try:
                import rich

                rich.print(f"[bold blue]🔎 Query ID[/bold blue]: {response.qid}")
                if isinstance(response, RetrievalEvaluatorResult):
                    rich.print(f"[bold blue]📜 Document ID[/bold blue]: {response.did}")
                elif response.agent_a and response.agent_b:
                    rich.print(
                        f"[bold blue] {response.agent_a:<18} [/bold blue] 🆚  "
                        f"[bold red] {response.agent_b}[/bold red]"
                    )
                else:
                    rich.print(f"[bold blue]🕵️ Agent[/bold blue]: {response.agent}")

                rich.print(f"[bold blue]Raw Answer[/bold blue]: {response.raw_answer}")
                rich.print(f"[bold blue]Parsed Answer[/bold blue]: {answer}")
                rich.print("")
                return
            except ImportError:
                logger.warning("Rich not installed. Using plain print")
        tqdm.write(f"Query ID: {response.qid}")
        if isinstance(response, RetrievalEvaluatorResult):
            tqdm.write(f"Document ID: {response.did}")
        elif response.agent_a and response.agent_b:
            tqdm.write(f"{response.agent_a} vs {response.agent_b}")
        else:
            tqdm.write(f"Agent: {response.agent}")
        tqdm.write(f"Raw Answer: {response.raw_answer}")
        tqdm.write(f"Parsed Answer: {answer}")
        tqdm.write("")

    @staticmethod
    def __dump_response_csv(
        answer_dict: dict[str, str], output_columns: list[str], output_file: str
    ):
        if not any(k in output_columns for k in answer_dict.keys()):
            raise ValueError(
                "No parsed answer fields are in the output columns. \n"
                f"Expected output columns: {output_columns}. \n"
                f"Answer fields: {answer_dict.keys()}"
            )

        if not os.path.isfile(output_file):
            logger.debug(f"Creating new file {output_file}")
            with open(output_file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=output_columns)
                writer.writeheader()
        with open(output_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=output_columns)
            writer.writerow(answer_dict)

    @staticmethod
    def __dump_response_jsonl(answer_dict: dict[str, str], output_file: str):
        with open(output_file, "a") as f:
            f.write(json.dumps(answer_dict) + "\n")

    @staticmethod
    def __dump_response_json(answer_dict: dict[str, str], output_file: str):
        """The file is a json-formatted list of dictionaries. Each dictionary is a response.
        If the file already exists, erase the final closing square bracket and add a comma before adding the new response.
        """
        if not os.path.isfile(output_file):
            logger.debug(f"Creating new file {output_file}")
            with open(output_file, "w") as f:
                f.write("[")
        with open(output_file, "r+") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 1:
                f.seek(f.tell() - 1)
                f.truncate()
                f.write(",")
            f.write(json.dumps(answer_dict) + "]")

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
        answer_dict = response.model_dump()
        if isinstance(answer_dict["answer"], dict):
            # flatten the dict into the main dict
            for k, v in answer_dict["answer"].items():
                answer_dict[k] = v
            del answer_dict["answer"]

        null_keys = {k for k in answer_dict.keys() if answer_dict[k] is None}
        unused_keys = (set(answer_dict.keys()) - set(output_columns)) & set(null_keys)
        for k in unused_keys:
            del answer_dict[k]
            # assert at least one of the new keys is in the output columns
        output_file = file if file else self.output_file
        if output_file.endswith(".csv"):
            self.__dump_response_csv(answer_dict, output_columns, output_file)
        elif output_file.endswith(".jsonl"):
            self.__dump_response_jsonl(answer_dict, output_file)
        elif output_file.endswith(".json"):
            self.__dump_response_json(answer_dict, output_file)
        else:
            logger.info(
                "Output file format not recognized. Dumping raw response in csv format."
            )
            self.__dump_response_csv(answer_dict, output_columns, output_file)

    def _process_answer(self, answer: str) -> Any:
        """Processes the LLM evaluator output into some serializable format"""
        if self.config.answer_format == AnswerFormat.JSON:
            return self.json_answer_parser(answer, self.config.scoring_key)
        if self.config.answer_format == AnswerFormat.MULTI_FIELD_JSON:
            assert isinstance(self.config.scoring_key, list)
            return self.json_answer_parser_multifields(answer, self.config.scoring_key)
        if self.config.answer_format == AnswerFormat.TEXT:
            return answer

    @staticmethod
    def _assemble_query(
        query: Query | str, query_metadata: Optional[dict[str, Any]] = None
    ) -> Query:
        if isinstance(query, str):
            query = Query(qid="<no_qid>", query=query)
        query.add_metadata(query_metadata)
        return query

    @staticmethod
    def _assemble_document(
        document: Document | str, doc_metadata: Optional[dict[str, Any]] = None
    ) -> Document:
        if isinstance(document, str):
            did = "<no_did>"
            if doc_metadata:
                valid_id_fields = ["did", "doc_id", "document_id", "id", "_id"]
                valid_id_fields = [f for f in valid_id_fields if f in doc_metadata]
                if valid_id_fields:
                    did = doc_metadata[valid_id_fields[0]]
            document = Document(did=did, text=document)
        document.add_metadata(doc_metadata)
        return document

    def _assemble_documents(
        self,
        documents: list[str] | list[Document],
        doc_metadata: Optional[list[dict[str, Any]]] = None,
    ) -> list[Document]:
        if doc_metadata:
            assert len(documents) == len(doc_metadata)
        assembled_docs = []
        for idx, (doc, metadata) in enumerate(
            zip(documents, doc_metadata or [None] * len(documents))
        ):
            if isinstance(doc, str) and not metadata:
                metadata = {"did": f"doc_{idx}>"}
            created_doc = self._assemble_document(doc, metadata)
            assembled_docs.append(created_doc)
        return assembled_docs

    @staticmethod
    def _assemble_answer(
        answer: AgentAnswer | str, answer_metadata: Optional[dict[str, Any]] = None
    ) -> AgentAnswer:
        if isinstance(answer, str):
            answer = AgentAnswer(agent="<no_agent>", text=answer)
        answer.add_metadata(answer_metadata)
        return answer
