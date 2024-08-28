"""Utils functions for RAGElo"""

import csv
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Union

from ragelo.types import AgentAnswer, Document, EvaluatorResult, PairwiseGame, Query


def infer_query_id_column(file_path: str) -> Union[str, None]:
    """Infer the column name with the query id from a CSV file."""
    possible_qid_columns = [
        "query_id",
        "qid",
        "question_id",
        "q_id",
        "queryid",
        "questionid",
    ]
    with open(file_path) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        if columns is not None:
            for col in possible_qid_columns:
                if col in columns:
                    return col
    return None


def infer_doc_id_column(file_path: str) -> Union[str, None]:
    """Infer the column name with the document id from a CSV file."""
    possible_did_columns = [
        "document_id",
        "doc_id",
        "did",
        "docid",
        "d_id",
        "documentid",
        "passageid",
        "pid",
        "p_id",
    ]
    with open(file_path) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        if columns is not None:
            for col in possible_did_columns:
                if col in columns:
                    return col
    return None


def load_queries_from_csv(
    queries_path: str,
    query_text_col: str = "query",
    query_id_col: Optional[str] = None,
    infer_metadata_fields: bool = True,
    metadata_fields: Optional[List[str]] = None,
) -> List[Query]:
    """Loads the queries from a CSV file and returns a dictionary with the queries.
        The CSV file should have a header with the columns 'query_id' and 'query'.
        The key is the query id and the value is the query object.

    Args:
        queries_path (str): Path to the CSV file with the queries.
        query_text_col (str): Name of the column with the query text. Defaults to 'query'.
        query_id_col (str): Name of the column with the query id. If not passed, try to infer from the csv.
            Uses the query_text_col as last resource.
        infer_metadata_fields (bool): Wether to infer the metadata fields in the csv.
        metadata_fields (Optional[List[str]]): The list of columns to be used as metadata.
            If infer_metadata_fields is True, will use all the fields if not explicitly set.
    Returns:
        List[Query]: List of queries.

    """
    queries = []
    read_queries = set()
    if not os.path.isfile(queries_path):
        raise FileNotFoundError(f"Queries file {queries_path} not found")
    if query_id_col is None:
        query_id_col = infer_query_id_column(queries_path)
    if query_id_col is None:
        query_id_col = query_text_col

    with open(queries_path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        if headers and infer_metadata_fields:
            if metadata_fields is None:
                metadata_fields = [
                    x for x in headers if x not in [query_id_col, query_text_col]
                ]
            else:
                # Use only the metadata fields that actually exist
                metadata_fields = [x for x in metadata_fields if x in headers]
                if not metadata_fields:
                    logging.warning(
                        "No metadata fields found in the csv. Ignoring metadata fields"
                    )
                    metadata_fields = None
        for line in reader:
            qid = line[query_id_col].strip()
            if qid in read_queries:
                logging.warning(f"Query {qid} already read. Skipping")
                continue
            query_text = line[query_text_col].strip()
            if metadata_fields is not None:
                extra_metadata = {k: v for k, v in line.items() if k in metadata_fields}
                queries.append(
                    Query(qid=qid, query=query_text, metadata=extra_metadata)
                )
            else:
                queries.append(Query(qid=qid, query=query_text))

            read_queries.add(qid)
    logging.info(f"Loaded {len(queries)} queries")

    return queries


def add_documents_from_csv(
    documents_file: str,
    queries_file: Optional[str] = None,
    queries: Optional[List[Query]] = None,
    query_id_col: str = "qid",
    document_id_col: str = "did",
    document_text_col: str = "document_text",
) -> List[Query]:
    """Loads a list of retrieved documents for each query. If queries is a string, will load the queries from the csv file.

    Args:
        documents_path (str): Path to the CSV file with the documents.
        queries (Optional[Dict[str, Query]]): Dictionary with the queries.
            If provided, only documents for the queries in the dictionary will be
            loaded. Defaults to None.
        query_id_col (str): Name of the column with the query id. Defaults to 'query_id'.
        document_id_col (str): Name of the column with the document id.
            Defaults to 'did'.
        document_text_col (str): Name of the column with the document text.
            Defaults to 'document_text'.

    Returns:
        List[Query]: A list of queries with the retrieved documents.

    """
    documents_read = 0
    if not os.path.isfile(documents_file):
        raise FileNotFoundError(f"Documents file {documents_file} not found")

    if queries is None and queries_file is None:
        raise ValueError("Either queries or queries_file must be provided")
    if queries is None:
        assert queries_file is not None  # Should not happen. but mypy complains
        queries = load_queries_from_csv(queries_file)
    queries_dict = {q.qid: q for q in queries}

    for line in csv.DictReader(open(documents_file)):
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
        queries_dict[qid].add_retrieved_doc(doc)
        documents_read += 1
    logging.info(f"Loaded {documents_read} documents")
    return list(queries_dict.values())


def add_documents_from_run_file(
    run_file_path: str,
    documents_path: str,
    queries_file: Optional[str] = None,
    queries: Optional[List[Query]] = None,
    document_id_col: str = "document_id",
    document_text_col: str = "document_text",
) -> List[Query]:
    """Loads documents based on a TREC-formatted run file and a CSV with the documents."""

    documents_read = 0
    if queries is None and queries_file is None:
        raise ValueError("Either queries or queries_file must be provided")
    if queries is None:
        assert queries_file is not None  # Should not happen. but mypy complains
        queries = load_queries_from_csv(queries_file)
    queries_dict = {q.qid: q for q in queries}

    if not os.path.isfile(run_file_path):
        raise FileNotFoundError(f"Run file {run_file_path} not found")
    if not os.path.isfile(documents_path):
        raise FileNotFoundError(f"Documents file {documents_path} not found")
    # A dictionary with all queries associated with each document
    query_per_doc = defaultdict(set)

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
            queries_dict[qid].add_retrieved_doc(Document(did=did, text=text))
            documents_read += 1
    logging.info(f"Loaded {documents_read} documents")
    return list(queries_dict.values())


def load_answers_from_multiple_csvs(
    answer_files: List[str],
    queries: Optional[Union[List[Query], str]] = None,
    query_text_col: str = "query",
    answer_text_col: str = "answer",
    query_id_col: Optional[str] = None,
):
    """Loads queries and answers from a list of CSVs with queries and answers generated by agents.
    We load all queries from all csvs first, and add the answers for all agents to the query objects.
    The agents will be named according to the csv file name.
    Args:
        answer_files (List[str]): A list of CSVs files with agent answers.
        queries (optional(Dict[str, Query] | str)): Either a dictionary with
            existing queries, a csv file with the queries or None. If none, will
            assume the queries also exist in the answers file.
        query_text_col (str): Name of the column with the query.
        answer_text_col (str): Name of the column with the agent's answer.
        query_id_col (str): Name of the column with the query id. If not defined, will try to infer it from the csv headers
    """
    queries = []
    queries_dict = {}

    for f in answer_files:
        if query_id_col is None:
            query_id_col = infer_query_id_column(f)
        if query_id_col is None:
            query_id_col = query_text_col
        p = Path(f)
        agent_name = p.stem

        for line in csv.DictReader(open(p)):
            qid = line[query_id_col]
            query_text = line[query_text_col].strip()
            if qid not in queries_dict:
                query = Query(qid=qid, query=query_text)
                queries_dict[qid] = query
                queries.append(query)
                logging.info(f'added query: "{qid}"')
            agent_answer = line[answer_text_col]
            answer = AgentAnswer(agent=agent_name, text=agent_answer)
            queries_dict[qid].add_agent_answer(answer)

    return queries


def add_answers_from_csv(
    answers_file: str,
    queries_file: Optional[str] = None,
    queries: Optional[List[Query]] = None,
    agent_col: str = "agent",
    answer_col: str = "answer",
    query_id_col: Optional[str] = None,
) -> List[Query]:
    """Loads all answers and agents from an answers CSV file.
    Args:
        answers_path (str): Path to the CSV file with the answers.
        queries (Dict[str, Query]): Dictionary with the queries.
        query_id_col (str): Name of the column with the query id. If not defined, will try to infer it from the csv headers
        agent_col (str): Name of the column with the agent. Defaults to 'agent'.
        answer_col (str): Name of the column with the answer. Defaults to 'answer'.
    Returns:
        List[Query]: A list of queries with the answers.
    """

    if not os.path.isfile(answers_file):
        raise FileNotFoundError(f"Answers file {answers_file} not found")

    if queries is None and queries_file is None:
        raise ValueError("Either queries or queries_file must be provided")
    if queries is None:
        assert queries_file is not None  # Should not happen. but mypy complains
        queries = load_queries_from_csv(queries_file)

    queries_dict = {q.qid: q for q in queries}
    query_id_col = query_id_col or infer_query_id_column(answers_file)
    for line in csv.DictReader(open(answers_file)):
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
        queries_dict[qid].add_agent_answer(answer)

    return list(queries_dict.values())


# TODO: Replace all this loading by JSON serialization of Pydantic models
def load_answer_evaluations_from_csv(
    evaluations_file: str,
) -> List[Query]:
    """Loads all evaluations produced by an answer evaluator from a CSV file."""

    queries_dict = {}
    for row in csv.DictReader(open(evaluations_file)):
        qid = row["qid"]
        agent_a = row["agent_a"]
        agent_b = row["agent_b"]
        raw_answer = row["raw_answer"]
        answer = row["answer"]

        if qid not in queries_dict:
            queries_dict[qid] = Query(qid=qid, query="<unknown>")
        query = queries_dict[qid]
        query.pairwise_games.append(
            PairwiseGame(
                agent_a_answer=AgentAnswer(agent=agent_a, text=""),
                agent_b_answer=AgentAnswer(agent=agent_b, text=""),
                evaluation=EvaluatorResult(
                    raw_answer=raw_answer, answer=answer, qid=qid
                ),
            )
        )
    queries = list(queries_dict.values())
    return queries
