"""Base model for dealing with document evaluators"""
import csv
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List, Type

from loguru import logger
from rich import print

from auto_eval.opeanai_client import OpenAiClient, set_credentials_from_file


class DocumentEvaluator:
    def __init__(
        self,
        query_path: str,
        documents_path: str,
        output_file: str,
        prompt_name: str,
        model_name: str = "gpt-4",
        credentials_file: str | None = None,
        print_answers: bool = False,
        force: bool = False,
    ):
        self.print_answers = print_answers
        self.force = force
        self.output_file = output_file
        self.queries = self._load_queries(query_path)
        self.documents = self._load_documents(documents_path)
        self.prompt = self._load_prompt(prompt_name)

        if credentials_file and os.path.isfile(credentials_file):
            set_credentials_from_file(credentials_file)

        self.openai_client = OpenAiClient(model=model_name)

    def _load_prompt(self, prompt_name: str) -> str:
        prompts_path = f"prompts/retrieval/{prompt_name}.txt"
        if not os.path.isfile(prompts_path):
            logger.exception(f"Prompts file {prompts_path} not found")
            raise FileNotFoundError

        with open(prompts_path) as f:
            prompt = f.read().strip()
        return prompt

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
        print(f"Loaded {len(queries)} queries")

        return queries

    def __title_overlap(self, new_doc: str, current_doc: str) -> int:
        size = 0
        min_len = min(len(new_doc), len(current_doc))
        while size < min_len and new_doc[size] == current_doc[size]:
            size += 1
        return size

    def _load_documents(self, documents_path: str) -> Dict[str, Dict[str, str]]:
        rows = defaultdict(dict)
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
                    f"Document {did} found more than once. trying to merge them by title"
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

    @abstractmethod
    def get_answers(self):
        pass


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
