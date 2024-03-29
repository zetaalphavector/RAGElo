"""Utils functions for RAGElo"""

import csv
import logging
import os
from collections import defaultdict
from typing import Optional

from ragelo.types import AgentAnswer, Document, Query


def load_queries_from_csv(
    queries_path: str, query_id_col: str = "query_id", query_text_col: str = "query"
) -> dict[str, Query]:
    """Loads the queries from a CSV file and returns a dictionary with the queries.
        The CSV file should have a header with the columns 'query_id' and 'query'.
        The key is the query id and the value is the query object.

    Args:
        queries_path (str): Path to the CSV file with the queries.
        query_id_col (str): Name of the column with the query id. Defaults to 'query_id'.
        query_text_col (str): Name of the column with the query text. Defaults to 'query'.
    Returns:
        dict[str, Query]: Dictionary with the queries.

    """
    queries = {}
    if not os.path.isfile(queries_path):
        raise FileNotFoundError(f"Queries file {queries_path} not found")

    for line in csv.DictReader(open(queries_path)):
        qid = line[query_id_col].strip()
        query_text = line[query_text_col].strip()
        queries[qid] = Query(qid, query_text)
    logging.info(f"Loaded {len(queries)} queries")

    return queries


def load_documents_from_file(
    documents_path: str,
    queries: dict[str, Query],
    query_id_col: str = "query_id",
    document_id_col: str = "doc_id",
    document_text_col: str = "document_text",
) -> dict[str, dict[str, Document]]:
    """Loads documents from a CSV and returns a dictionary with their content.

    Args:
        documents_path (str): Path to the CSV file with the documents.
        queries (Optional[dict[str, Query]]): Dictionary with the queries.
            If provided, only documents for the queries in the dictionary will be
            loaded. Defaults to None.
        query_id_col (str): Name of the column with the query id. Defaults to 'query_id'.
        document_id_col (str): Name of the column with the document id.
            Defaults to 'doc_id'.
        document_text_col (str): Name of the column with the document text.
            Defaults to 'document_text'.

    Returns:
        dict[str, dict[str, Document]]: Dictionary mapping a qid to its documents.

    """
    documents: dict[str, dict[str, Document]] = {}
    documents_read = 0
    if not os.path.isfile(documents_path):
        raise FileNotFoundError(f"Documents file {documents_path} not found")

    for line in csv.DictReader(open(documents_path)):
        qid = line[query_id_col].strip()
        did = line[document_id_col].strip()
        text = line[document_text_col].strip()
        if queries and qid not in queries.keys():
            continue
        if qid not in documents:
            documents[qid] = {}
        documents[qid][did] = Document(queries[qid], did, text)
        documents_read += 1
    logging.info(f"Loaded {documents_read} documents")
    return documents


def load_documents_from_run_file(
    run_file_path: str,
    documents_path: str,
    queries: Optional[dict[str, Query]] = None,
    document_id_col: str = "document_id",
    document_text_col: str = "document_text",
) -> dict[str, dict[str, Document]]:
    """Loads documents based on a TREC-formatted run file and a CSV with the documents."""
    documents: dict[str, dict[str, Document]] = defaultdict(dict)
    documents_read = 0
    if not os.path.isfile(run_file_path):
        raise FileNotFoundError(f"Run file {run_file_path} not found")
    if not os.path.isfile(documents_path):
        raise FileNotFoundError(f"Documents file {documents_path} not found")
    # A dictionary with all queries associated with each document
    query_per_doc = defaultdict(set)

    for line in open(run_file_path, "r"):
        qid, _, did, _, _, _ = line.strip().split()
        if queries and qid not in queries:
            continue
        query_per_doc[did].add(qid)

    for line in csv.DictReader(open(documents_path)):
        did = line[document_id_col].strip()
        if did not in query_per_doc:
            continue
        text = line[document_text_col].strip()
        for qid in query_per_doc[did]:
            documents[qid][did] = Document(qid, did, text)
            documents_read += 1
    logging.info(f"Loaded {documents_read} documents")

    return documents


def load_answers_and_agents_from_csv(
    answers_path: str,
    queries: dict[str, Query],
    query_id_col: str = "query_id",
    agent_col: str = "agent",
    answer_col: str = "answer",
) -> dict[str, list[AgentAnswer]]:
    # ) -> dict[str, dict[str, str]]:
    """Loads all answers and agents from an answers CSV file.
    Args:
        answers_path (str): Path to the CSV file with the answers.
        queries (dict[str, Query]): Dictionary with the queries.
        query_id_col (str): Name of the column with the query id. Defaults to 'query_id'.
        agent_col (str): Name of the column with the agent. Defaults to 'agent'.
        answer_col (str): Name of the column with the answer. Defaults to 'answer'.
    Returns:
        dict[str, list[AgentAnswer]]: Dictionary with the answers for each query.
    """
    answers: dict[str, list[AgentAnswer]] = defaultdict(list)
    for line in csv.DictReader(open(answers_path)):
        qid = line[query_id_col]
        if qid not in queries:
            raise ValueError(f"Unknown query id {qid}")
        query = queries[qid]
        agent = line[agent_col].strip()
        answer = line[answer_col].strip()
        answers[qid].append(AgentAnswer(query, agent, answer))

    return dict(answers)
