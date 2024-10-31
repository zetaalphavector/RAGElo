"""A Queries object contain multiple user queries that can be evaluated at once. This is also useful for aggregating the performance of a pipeline over multiple queries."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

from pydantic import Field

from ragelo.logger import logger
from ragelo.types.evaluables import Document, Evaluable
from ragelo.types.pydantic_models import BaseModel, validator
from ragelo.types.query import Query
from ragelo.types.results import AnswerEvaluatorResult, RetrievalEvaluatorResult


class Queries(BaseModel):
    """A collection of queries and their retrieved documents and agent answers.
    Attributes:
        queries: A dictionary of Query objects, where the key is the query id.
    """

    queries: dict[str, Query] = Field(default_factory=dict)
    experiment_id: str | None = None
    cache_path: str | None = None
    evaluables_cache_path: str | None = None
    save_cache: bool = True
    csv_path: str | None = None
    clear_evaluations: bool = False
    csv_query_text_col: str = "query"
    csv_query_id_col: str | None = None
    csv_infer_metadata_fields: bool = True
    csv_metadata_fields: list[str] | None = None

    def __init__(self, **data):
        super().__init__(**data)
        if not self.experiment_id:
            self.experiment_id = str(int(datetime.now(timezone.utc).timestamp()))
        if self.cache_path is None:
            os.makedirs("cache", exist_ok=True)
            self.cache_path = f"cache/{self.experiment_id}.json"
        if self.evaluables_cache_path is None:
            self.evaluables_cache_path = self.cache_path.replace(
                ".json", "_evaluables.json"
            )
        # Check where to load the data from. If the cache file exists or the csv_file exists, load from there. If the queries was also set, make sure they match.
        if self.csv_path:
            csv_queries = self.load_from_csv()
            if len(self.queries) > 0 and set(csv_queries.keys()) != set(
                self.queries.keys()
            ):
                raise ValueError(
                    "Queries loaded from the CSV file do not match the queries passed"
                )
            self.queries = csv_queries
        elif self.cache_path and os.path.isfile(self.cache_path):
            cache_queries = self.load_from_cache()
            if len(self.queries) > 0 and set(cache_queries.keys()) != set(
                self.queries.keys()
            ):
                raise ValueError(
                    "Queries loaded from the cache file do not match the queries passed"
                )
            self.queries = cache_queries
        if self.clear_evaluations:
            self.clear_all_evaluations()

    def clear_all_evaluations(self):
        logger.warning(f"Clearing all evaluations for {len(self)} queries")
        doc_eval_count = 0
        game_eval_count = 0
        answer_eval_count = 0
        for query in self.queries.values():
            for doc in query.retrieved_docs.values():
                if doc.evaluation is not None:
                    doc_eval_count += 1
                doc.evaluation = None
            for game in query.pairwise_games:
                if game.evaluation is not None:
                    game_eval_count += 1
                game.evaluation = None
            for answer in query.answers.values():
                if answer.evaluation is not None:
                    answer_eval_count += 1
                answer.evaluation = None
        logger.info(
            f"Cleared {doc_eval_count} document evaluations, {game_eval_count} game evaluations, and {answer_eval_count} answer evaluations"
        )
        # clear the evaluables cache file

        if self.evaluables_cache_path and os.path.isfile(self.evaluables_cache_path):
            with open(self.evaluables_cache_path, "w") as f:
                pass
        self.dump()

    def load_from_csv(self) -> dict[str, Query]:
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
        if self.csv_path is None:
            raise ValueError("csv_path not set. Cannot load queries")
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV file {self.csv_path} not found")
        if self.csv_query_id_col is None:
            self.csv_query_id_col = self._infer_query_id_column(self.csv_path)
        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            if headers and self.csv_infer_metadata_fields:
                if self.csv_metadata_fields is None:
                    self.csv_metadata_fields = [
                        x
                        for x in headers
                        if x not in [self.csv_query_id_col, self.csv_query_text_col]
                    ]
                else:
                    # Use only the metadata fields that actually exist
                    self.csv_metadata_fields = [
                        x for x in self.csv_metadata_fields if x in headers
                    ]
                    if not self.csv_metadata_fields:
                        logger.warning(
                            "No metadata fields found in the csv. Ignoring metadata fields"
                        )
                        self.csv_metadata_fields = None
            for idx, row in enumerate(reader):
                qid = row.get(self.csv_query_id_col) or f"query_{idx}"
                if qid in read_queries:
                    logger.warning(f"Query with ID {qid} already read. Skipping")
                    continue
                query_text = row[self.csv_query_text_col].strip()
                metadata = None
                if self.csv_metadata_fields is not None:
                    metadata = {
                        k: v for k, v in row.items() if k in self.csv_metadata_fields
                    }
                queries[qid] = Query(qid=qid, query=query_text, metadata=metadata)
                read_queries.add(qid)
        return queries

    def load_from_cache(self):
        queries: dict[str, Query] = {}
        if not self.cache_path:
            raise ValueError("Cache path not set. Cannot load queries")
        if not os.path.isfile(self.cache_path):
            raise FileNotFoundError(f"Cache file {self.cache_path} not found")
        with open(self.cache_path) as f:
            queries = {k: Query(**v) for k, v in json.load(f)["queries"].items()}
        if self.evaluables_cache_path is None:
            self.evaluables_cache_path = self.cache_path.replace(
                ".json", "_evaluables.json"
            )
            with open(self.evaluables_cache_path, "w") as f:
                pass
        for line in open(self.evaluables_cache_path):
            evaluable = json.loads(line)
            evaluable_type = list(evaluable.keys())[0]
            if evaluable_type == "answer":
                result = AnswerEvaluatorResult(**evaluable["answer"])
                if (
                    result.qid not in queries
                    or result.agent not in queries[result.qid].answers
                ):
                    continue
                queries[result.qid].answers[result.agent].evaluation = result
            elif evaluable_type == "retrieval":
                result = RetrievalEvaluatorResult(**evaluable["retrieval"])
                if (
                    result.qid not in queries
                    or result.did not in queries[result.qid].retrieved_docs
                ):
                    continue
                queries[result.qid].retrieved_docs[result.did].evaluation = result
        return queries

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

    def add_evaluation(
        self, evaluation: RetrievalEvaluatorResult | AnswerEvaluatorResult
    ):
        if isinstance(evaluation, RetrievalEvaluatorResult):
            self.add_retrieval_evaluation(evaluation)
        elif isinstance(evaluation, AnswerEvaluatorResult):
            self.add_answer_evaluation(evaluation)
        else:
            raise ValueError(
                f"Cannot add evaluation of type {type(evaluation)} to queries"
            )

    def add_retrieval_evaluation(self, evaluation: RetrievalEvaluatorResult):
        qid = evaluation.qid
        if qid not in self.queries:
            raise ValueError(
                f"Trying to add retrieval evaluation for non-existing query {qid}"
            )
        did = evaluation.did
        if did not in self.queries[qid].retrieved_docs:
            raise ValueError(
                f"Trying to add evaluation for non-retrieved document {did} in query {qid}"
            )
        if self.queries[qid].retrieved_docs[did].evaluation is not None:
            logger.warning(
                f"Query {qid} already has an evaluation for document {did}. Overwriting."
            )
        self.queries[qid].retrieved_docs[did].evaluation = evaluation
        self.save_result(evaluation)

    def add_answer_evaluation(self, evaluation: AnswerEvaluatorResult):
        qid = evaluation.qid
        if qid not in self.queries:
            raise ValueError(
                f"Trying to add ANswer evaluation for non-existing query {qid}"
            )
        if evaluation.pairwise:
            agent_a = evaluation.agent_a
            agent_b = evaluation.agent_b
            for game in self.queries[qid].pairwise_games:
                if (
                    game.agent_a_answer.agent == agent_a
                    and game.agent_b_answer.agent == agent_b
                ):
                    if game.evaluation is not None:
                        logger.warning(
                            f"Query {qid} already has an evaluation for agents {agent_a} and {agent_b}. Overwriting."
                        )
                    game.evaluation = evaluation
                    self.save_result(evaluation)
                    return
        agent = evaluation.agent
        if agent not in self.queries[qid].answers:
            raise ValueError(
                f"Trying to add evaluation for non-existing agent {agent} in query {qid}"
            )
        if self.queries[qid].answers[agent].evaluation is not None:
            logger.warning(
                f"Query {qid} already has an evaluation for agent {agent}. Overwriting."
            )
        self.queries[qid].answers[agent].evaluation = evaluation
        self.save_result(evaluation)

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
        if not self.save_cache:
            return
        if self.cache_path is None:
            raise ValueError("Cache path not set. Cannot dump queries")
        with open(self.cache_path, "w") as f:
            json.dump(self.model_dump(), f)

    def save_result(self, evaluable: AnswerEvaluatorResult | RetrievalEvaluatorResult):
        if not self.save_cache:
            return
        if self.evaluables_cache_path is None:
            raise ValueError("Evaluables cache path not set. Cannot dump evaluables")
        with open(self.evaluables_cache_path, "a+") as f:
            if isinstance(evaluable, AnswerEvaluatorResult):
                evaluable_type = "answer"
            elif isinstance(evaluable, RetrievalEvaluatorResult):
                evaluable_type = "retrieval"
            else:
                raise ValueError(
                    f"Cannot save evaluation of type {type(evaluable)} to cache"
                )
            f.write(
                json.dumps({evaluable_type: evaluable.model_dump()}, indent=2) + "\n"
            )

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

    def keys(self):
        return self.queries.keys()
