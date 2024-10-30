"""A Queries object contain multiple user queries that can be evaluated at once. This is also useful for aggregating the performance of a pipeline over multiple queries."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator
from ragelo.logger import logger
from ragelo.types.evaluables import Document
from ragelo.types.pydantic_models import BaseModel, validator
from ragelo.types.query import Query


class Queries(BaseModel):
    """A collection of queries and their retrieved documents and agent answers.
    Attributes:
        queries: A dictionary of Query objects, where the key is the query id.
    """

    queries: dict[str, Query] = Field(default_factory=dict)
    experiment_id: str | None = None
    cache_path: str | None = None

    @validator
    @classmethod
    def add_cache_path(cls, v):
        experiment_id = v.get("experiment_id")
        if experiment_id is None:
            experiment_id = str(int(datetime.now(timezone.utc).timestamp()))
        cache_path = v.get("cache_path")
        if cache_path is None:
            os.makedirs("cache", exist_ok=True)
            cache_path = f"cache/{experiment_id}.json"
        v["cache_path"] = cache_path
        return v

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        query_text_col: str = "query",
        query_id_col: str | None = None,
        infer_metadata_fields: bool = True,
        metadata_fields: list[str] | None = None,
        experiment_id: str | None = None,
        cache_path: str | None = None,
    ):
        """Loads a list of queries from a CSV file.
            The CSV file should have a header and at least one column (specified by query_text_col) with the query text.
            If no query_id_col is defined, we will first try to infer what column should be used as id from a list of pre-set values. If it fails, the queries will have ids assigned to them according to their order in the csv file.

        Args:
            queries_path (str): Path to the CSV file with the queries.
            query_text_col (str): Name of the column with the query text. Defaults to 'query'.
            query_id_col (str): Name of the column with the query id. If not passed, try to infer from the csv.
                Uses the query_text_col as last resource.
            infer_metadata_fields (bool): Wether to infer the metadata fields in the csv.
            metadata_fields (Optional[List[str]]): The list of columns to be used as metadata.
                If infer_metadata_fields is True, will use all the fields if not explicitly set.
            experiment_id (str): The id of the experiment. If not passed, a new one will be created with the current timestamp.
            cache_path (str): The path to the cache file. If not passed, a new one will be created with the experiment_id.
        """
        queries: dict[str, Query] = {}
        read_queries = set()
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Queries file {csv_path} not found")
        if query_id_col is None:
            query_id_col = cls._infer_query_id_column(csv_path)
        with open(csv_path) as f:
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
                        logger.warning(
                            "No metadata fields found in the csv. Ignoring metadata fields"
                        )
                        metadata_fields = None
            for idx, row in enumerate(reader):
                qid = row.get(query_id_col) or f"query_{idx}"
                if qid in read_queries:
                    logger.warning(f"Query with ID {qid} already read. Skipping")
                    continue
                query_text = row[query_text_col].strip()
                metadata = None
                if metadata_fields is not None:
                    metadata = {k: v for k, v in row.items() if k in metadata_fields}
                queries[qid] = Query(qid=qid, query=query_text, metadata=metadata)
                read_queries.add(qid)
        logger.info(f"Loaded {len(queries)} queries")
        return cls(queries=queries, experiment_id=experiment_id, cache_path=cache_path)

    @classmethod
    def from_cache(cls, cache_path: str):
        with open(cache_path) as f:
            return cls(**json.load(f))

    def evaluate_with_retrieval_evaluator(
        self, retrieval_evaluator: BaseRetrievalEvaluator
    ):
        retrieval_evaluator.batch_evaluate(list(self.queries.values()))
        self.dump()

    def add_retrieved_docs_from_runfile(
        self,
        run_file_path: str,
        corpus: dict[str, Document] | dict[str, str],
        top_k: int | None = None,
    ):
        """Adds the retrieved documents from a run file to the queries.
        A run file has, traditionally, the following format:
            qid, Q0, docid, rank, score, run_name
        The run name will be considered as the agent that retrieved the documents. The second column is usually ignored.
        Args:
            run_file_path (str): Path to the run file.
            corpus (dict[str, Document] | dict[str, str]): A dictionary with the documents.
                If a dictionary with just docid and text is provided, we will use it as a corpus.
                Otherwise, we will assume it's a dictionary with docid as key and Document object as value.
            top_k (int): The maximum number of documents to add for each query. Assumes that the run file is ordered by rank.
        """
        documents_read = set()
        missing_docs = set()
        docs_per_query = {}
        warned_queries = set()

        if isinstance(list(corpus.values())[0], str):
            corpus = {k: Document(did=k, text=str(v)) for k, v in corpus.items()}

        if not os.path.isfile(run_file_path):
            raise FileNotFoundError(f"Run file {run_file_path} not found")
        for line in open(run_file_path):
            qid, _, did, _, score, agent = line.strip().split()
            try:
                doc = corpus[did]
            except KeyError:
                missing_docs.add(did)
                continue
            if qid not in self.queries:
                if qid not in warned_queries:
                    warned_queries.add(qid)
                    logger.warning(f"Query {qid} not found in queries. Skipping")
                continue
            if qid not in docs_per_query:
                docs_per_query[qid] = 0
            if docs_per_query[qid] < top_k:
                self.queries[qid].add_retrieved_doc(
                    doc, agent=agent, score=float(score)
                )
                docs_per_query[qid] += 1
                documents_read.add(did)
        logger.info(
            f"Loaded {len(documents_read)} documents. {len(missing_docs)} missing docs"
        )

    def get_qrels(
        self, relevance_key: str | None = "answer", relevance_threshold: int = 0
    ) -> dict[str, dict[str, int]]:
        qrels = {}
        for qid, query in self.queries.items():
            qrels[qid] = query.get_qrels(relevance_key, relevance_threshold)
        return qrels

    def get_runs(
        self, agents: list[str] | None = None
    ) -> dict[str, dict[str, dict[str, float]]]:
        runs_by_agent = {}
        for query in self.queries.values():
            runs = query.get_runs(agents)
            for agent, runs in runs.items():
                runs_by_agent[agent] = runs_by_agent.get(agent, {}) | runs
        return runs_by_agent

    def dump(self):
        if self.cache_path is None:
            raise ValueError("Cache path not set. Cannot dump queries")
        with open(self.cache_path, "w") as f:
            json.dump(self.model_dump(), f)

    def __len__(self):
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries.values())

    def __getitem__(self, key: str) -> Query:
        return self.queries[key]

    def evaluate_retrieval(
        self,
        metrics: list[str] = ["Precision@10", "nDCG@10", "Judged@10"],
    ):
        try:
            import ir_measures
            from ir_measures import parse_measure
        except ImportError:
            raise ImportError(
                "ir_measures is not installed. Please install it with `pip install ir-measures`"
            )
        qrels = self.get_qrels()
        runs = self.get_runs()
        measures = []
        for metric in metrics:
            try:
                measure = parse_measure(metric)
            except NameError:
                valid_metrics = list(ir_measures.measures.registry.keys())
                raise ValueError(
                    f"Metric {metric} not found. Valid metrics are: {valid_metrics}"
                )
            measures.append(measure)
        metrics = [str(m) for m in measures]
        results = {}
        for agent, runs in self.get_runs().items():
            # Transform the keys of the results back to strings
            results[agent] = {
                str(k): v
                for k, v in ir_measures.calc_aggregate(measures, qrels, runs).items()  # type: ignore
            }

        key_metric = metrics[0]
        max_agent_len = max([len(agent) for agent in results.keys()]) + 3
        max_metric_len = max([len(metric) for metric in metrics])
        sorted_agents = sorted(
            results.items(), key=lambda x: x[1][key_metric], reverse=True
        )
        try:
            import rich

            rich.print("---[bold cyan] Retrieval Scores [/bold cyan]---")
            header = f"[bold magenta]{'Agent Name':<{max_agent_len}}"
            header += "\t".join([f"{m:<{max_metric_len}}" for m in metrics])
            header += "[/bold magenta]"
            rich.print(f"[bold cyan]{header}[/bold cyan]")
            for agent, scores in sorted_agents:
                row = f"[bold white]{agent:<{max_agent_len}}[/bold white]"
                row += "\t".join(
                    [f"{scores[metric]:<{max_metric_len},.4f}" for metric in metrics]
                )
                rich.print(row)

        except ImportError:
            logger.warning("Rich not installed. Using plain print")
            print(results)
        return results

    def keys(self):
        return self.queries.keys()

    @staticmethod
    def _infer_query_id_column(file_path: str) -> str | None:
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
