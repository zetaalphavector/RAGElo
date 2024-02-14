"""A Evaluator is a class that receives some input (e.g. a document and a query or a pair of answers) and returns a score or a label."""

import csv
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.types import Document, Query
from ragelo.types.configurations import EvaluatorConfig


class BaseEvaluator(ABC):
    @abstractmethod
    def __init__(
        self,
        queries: List[Query],
        llm_provider: BaseLLMProvider,
        config: EvaluatorConfig,
    ):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        """Runs the evaluator for all loaded samples."""
        raise NotImplementedError

    @staticmethod
    def load_queries(queries_path: str) -> Dict[str, Query]:
        """Loads the queries from a CSV file and returns a dictionary with the queries.
        The key is the query id and the value is the query object."""
        queries = {}
        if not os.path.isfile(queries_path):
            raise FileNotFoundError(f"Queries file {queries_path} not found")

        for line in csv.reader(open(queries_path)):
            if "query" in line:
                continue
            qid, query = line
            queries[qid] = Query(qid, query)
        logging.info(f"Loaded {len(queries)} queries")

        return queries

    @staticmethod
    def load_documents(
        documents_path: str, queries: Optional[List[Query]] = None
    ) -> Dict[str, Dict[str, Document]]:
        documents = {}
        documents_read = 0
        if not os.path.isfile(documents_path):
            raise FileNotFoundError(f"Documents file {documents_path} not found")

        for line in csv.reader(open(documents_path)):
            if "query_id" in line:
                continue

            qid, did, text = line
            if queries and qid not in queries:
                continue
            if qid not in documents:
                documents[qid] = {}
            documents[qid][did] = Document(qid, did, text)
            documents_read += 1
        logging.info(f"Loaded {documents_read} documents")
        return documents