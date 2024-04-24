"""Utils functions for RAGElo"""

import csv
import logging
import os
from collections import defaultdict

from ragelo.types.types import AgentAnswer, Document, Query


def load_queries_from_csv(
    queries_path: str, query_id_col: str = "qid", query_text_col: str = "query"
) -> list[Query]:
    """Loads the queries from a CSV file and returns a dictionary with the queries.
        The CSV file should have a header with the columns 'query_id' and 'query'.
        The key is the query id and the value is the query object.

    Args:
        queries_path (str): Path to the CSV file with the queries.
        query_id_col (str): Name of the column with the query id. Defaults to 'query_id'.
        query_text_col (str): Name of the column with the query text. Defaults to 'query'.
    Returns:
        list[Query]: List of queries.

    """
    queries = []
    read_queries = set()
    if not os.path.isfile(queries_path):
        raise FileNotFoundError(f"Queries file {queries_path} not found")
    with open(queries_path) as f:
        reader = csv.DictReader(f)
        for line in reader:
            qid = line[query_id_col].strip()
            query_text = line[query_text_col].strip()
            extra_metadata = {
                k: v for k, v in line.items() if k not in [query_id_col, query_text_col]
            }
            if qid in read_queries:
                logging.warning(f"Query {qid} already read. Skipping")
                continue
            queries.append(
                Query(qid=qid, query=query_text, metadata=extra_metadata or None)
            )
            read_queries.add(qid)
    logging.info(f"Loaded {len(queries)} queries")

    return queries


def load_retrieved_docs_from_csv(
    documents_path: str,
    queries: list[Query] | str,
    query_id_col: str = "qid",
    document_id_col: str = "did",
    document_text_col: str = "document_text",
) -> list[Query]:
    """Loads a list of retrieved documents for each query.

    Args:
        documents_path (str): Path to the CSV file with the documents.
        queries (Optional[dict[str, Query]]): Dictionary with the queries.
            If provided, only documents for the queries in the dictionary will be
            loaded. Defaults to None.
        query_id_col (str): Name of the column with the query id. Defaults to 'query_id'.
        document_id_col (str): Name of the column with the document id.
            Defaults to 'did'.
        document_text_col (str): Name of the column with the document text.
            Defaults to 'document_text'.

    Returns:
        list[Query]: A list of queries with the retrieved documents.

    """
    documents_read = 0
    if not os.path.isfile(documents_path):
        raise FileNotFoundError(f"Documents file {documents_path} not found")
    if isinstance(queries, str):
        queries = load_queries_from_csv(queries)
    queries_dict = {q.qid: q for q in queries}

    for line in csv.DictReader(open(documents_path)):
        qid = line[query_id_col].strip()
        did = line[document_id_col].strip()
        text = line[document_text_col].strip()
        extra_metadata = {
            k: v
            for k, v in line.items()
            if k not in [query_id_col, document_id_col, document_text_col]
        }
        if qid not in queries_dict:
            logging.info(f"Query {qid} not in the provided queries. Skipping")
            continue
        doc = Document(did=did, text=text, metadata=extra_metadata or None)
        queries_dict[qid].retrieved_docs.append(doc)
        documents_read += 1
    logging.info(f"Loaded {documents_read} documents")
    return list(queries_dict.values())


def load_retrieved_docs_from_run_file(
    run_file_path: str,
    documents_path: str,
    queries: list[Query] | str,
    document_id_col: str = "document_id",
    document_text_col: str = "document_text",
) -> list[Query]:
    """Loads documents based on a TREC-formatted run file and a CSV with the documents."""

    documents_read = 0
    if not os.path.isfile(run_file_path):
        raise FileNotFoundError(f"Run file {run_file_path} not found")
    if not os.path.isfile(documents_path):
        raise FileNotFoundError(f"Documents file {documents_path} not found")
    # A dictionary with all queries associated with each document
    query_per_doc = defaultdict(set)
    if isinstance(queries, str):
        queries = load_queries_from_csv(queries)
    queries_dict = {q.qid: q for q in queries}

    for _line in open(run_file_path, "r"):
        qid, _, did, _, _, _ = _line.strip().split()
        if qid not in queries_dict:
            logging.info(f"Query {qid} not in the provided queries. Skipping")
            continue
        query_per_doc[did].add(qid)

    for line in csv.DictReader(open(documents_path)):
        did = line[document_id_col].strip()
        if did not in query_per_doc:
            continue
        text = line[document_text_col].strip()
        for qid in query_per_doc[did]:
            queries_dict[qid].retrieved_docs.append(Document(did=did, text=text))
            documents_read += 1
    logging.info(f"Loaded {documents_read} documents")
    return list(queries_dict.values())


def load_answers_from_csv(
    answers_path: str,
    queries: list[Query] | str,
    query_id_col: str = "qid",
    agent_col: str = "agent",
    answer_col: str = "answer",
) -> list[Query]:
    """Loads all answers and agents from an answers CSV file.
    Args:
        answers_path (str): Path to the CSV file with the answers.
        queries (dict[str, Query]): Dictionary with the queries.
        query_id_col (str): Name of the column with the query id. Defaults to 'query_id'.
        agent_col (str): Name of the column with the agent. Defaults to 'agent'.
        answer_col (str): Name of the column with the answer. Defaults to 'answer'.
    Returns:
        list[Query]: A list of queries with the answers.
    """

    if not os.path.isfile(answers_path):
        raise FileNotFoundError(f"Answers file {answers_path} not found")
    if isinstance(queries, str):
        queries = load_queries_from_csv(queries)
    queries_dict = {q.qid: q for q in queries}

    for line in csv.DictReader(open(answers_path)):
        qid = line[query_id_col]
        if qid not in queries_dict:
            raise ValueError(f"Unknown query id {qid}")
        agent = line[agent_col].strip()
        answer = line[answer_col].strip()
        extra_metadata = {
            k: v
            for k, v in line.items()
            if k not in [query_id_col, agent_col, answer_col]
        }
        answer = AgentAnswer(agent=agent, text=answer, metadata=extra_metadata or None)
        queries_dict[qid].answers.append(answer)

    return list(queries_dict.values())
