"""A Queries object contain multiple user queries that can be evaluated at once. This is also useful for aggregating the performance of a pipeline over multiple queries."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

from pydantic import Field

from ragelo.logger import logger
from ragelo.types.evaluables import Document
from ragelo.types.pydantic_models import BaseModel, validator
from ragelo.types.query import Query
from ragelo.types.results import (
    AnswerEvaluatorResult,
    EloTournamentResult,
    RetrievalEvaluatorResult,
)


class Queries(BaseModel):
    """
    A class to manage and evaluate a collection of queries.
    Attributes:
        queries (dict[str, Query]): A dictionary of queries.
        experiment_id (str | None): The ID of the experiment.
        cache_path (str | None): The path to the cache file.
        results_cache_path (str | None): The path to the results cache file.
        save_cache (bool): Whether to save the cache to disk.
        csv_path (str | None): The path to the CSV file containing queries.
        clear_evaluations (bool): Whether to clear all evaluations.
        csv_query_text_col (str): The column name for query text in the CSV file.
        csv_query_id_col (str | None): The column name for query ID in the CSV file.
        csv_infer_metadata_fields (bool): Whether to infer metadata fields from the CSV file.
        csv_metadata_fields (list[str] | None): The list of metadata fields in the CSV file.
    Methods:
        __init__(**data): Initializes the Queries object.
        add_retrieved_docs_from_runfile(run_file_path: str, corpus: dict[str, Document] | dict[str, str], top_k: int | None = None): Adds retrieved documents from a run file to the queries.
        add_evaluation(evaluation: RetrievalEvaluatorResult | AnswerEvaluatorResult, should_save: bool = True): Adds an evaluation to the queries.
        add_retrieval_evaluation(evaluation: RetrievalEvaluatorResult): Adds a retrieval evaluation to the queries.
        add_answer_evaluation(evaluation: AnswerEvaluatorResult): Adds an answer evaluation to the queries.
        get_qrels(relevance_key: str | None = "answer", relevance_threshold: int = 0) -> dict[str, dict[str, int]]: Gets the qrels for the queries.
        get_runs(agents: list[str] | None = None) -> dict[str, dict[str, dict[str, float]]]: Gets the runs for the queries.
        persist_on_disk(): Saves the queries to disk.
        save_result(result: AnswerEvaluatorResult | RetrievalEvaluatorResult): Saves an evaluation result to disk.
        __len__(): Returns the number of queries.
        __iter__(): Returns an iterator over the queries.
        __getitem__(key: str) -> Query: Gets a query by its key.
        evaluate_retrieval(metrics: list[str] = ["Precision@10", "nDCG@10", "Judged@10"], relevance_threshold: int = 0): Evaluates the retrieval performance of the queries.
        _infer_query_id_column(file_path: str) -> str | None: Infers the column name for query ID from a CSV file.
        keys(): Returns the keys of the queries.
    """

    queries: dict[str, Query] = Field(default_factory=dict)
    experiment_id: str | None = None
    cache_path: str | None = None
    results_cache_path: str | None = None
    save_cache: bool = True
    csv_path: str | None = None
    clear_evaluations: bool = False
    csv_query_text_col: str = "query"
    csv_query_id_col: str | None = None
    csv_infer_metadata_fields: bool = True
    csv_metadata_fields: list[str] | None = None
    elo_tournaments: list[EloTournamentResult] = Field(default_factory=list)

    def __init__(self, **data):
        """
        Initialize the query object with optional data and set up paths and caches.

        The initialization process involves setting up paths and loading queries from different sources:
        1. If `csv_path` is provided, queries are loaded from the specified CSV file.
        2. If `csv_path` is not provided but `cache_path` exists and points to a valid file, queries are loaded from the cache file.
        3. If both `csv_path` and `cache_path` are provided, the method ensures that the queries loaded from the CSV file match those in the cache file.
        The method prioritizes loading queries from the CSV file over the cache file. If neither source is available, it relies on the `queries` attribute if it has been set.
        Additionally, if `results_cache_path` points to an existing file, results are loaded from the cache. If `clear_evaluations` is set to True, all evaluations are cleared.
        Finally, the state is persisted on disk.

        Keyword Args:
            **data: Arbitrary keyword arguments containing initialization data.
        Raises:
            ValueError: If the queries loaded from the CSV file or cache file do not match the queries passed.
        Attributes:
            experiment_id (str): Unique identifier for the experiment. Defaults to the current UTC timestamp if not provided.
            cache_path (str): Path to the cache file. Defaults to "cache/{experiment_id}.json" if not provided.
            results_cache_path (str): Path to the results cache file. Defaults to "cache/{experiment_id}_results.jsonl" if not provided.
            csv_path (str): Path to the CSV file containing queries. Optional.
            queries (dict): Dictionary of queries. Loaded from CSV or cache file if not provided.
            clear_evaluations (bool): Flag to clear evaluations. Optional.
        """

        super().__init__(**data)
        if not self.experiment_id:
            self.experiment_id = str(int(datetime.now(timezone.utc).timestamp()))
        if self.cache_path is None:
            os.makedirs("cache", exist_ok=True)
            self.cache_path = f"cache/{self.experiment_id}.json"
        if self.results_cache_path is None:
            self.results_cache_path = self.cache_path.replace(".json", "_results.jsonl")
        # Check where to load the data from. If the cache file exists or the csv_file exists, load from there. If the queries was also set, make sure they match.
        if self.csv_path:
            csv_queries = self._read_queries_csv()
            if len(self.queries) > 0 and set(csv_queries.keys()) != set(
                self.queries.keys()
            ):
                raise ValueError(
                    "Queries loaded from the CSV file do not match the queries passed"
                )
            self.queries = csv_queries
        elif self.cache_path and os.path.isfile(self.cache_path):
            cache_queries = self._load_cached_queries()
            if len(self.queries) > 0 and set(cache_queries.keys()) != set(
                self.queries.keys()
            ):
                raise ValueError(
                    "Queries loaded from the cache file do not match the queries passed"
                )
            self.queries = cache_queries
        if os.path.isfile(self.results_cache_path):
            self._load_results_from_cache()
        if self.clear_evaluations:
            self._clear_all_evaluations()
        self.persist_on_disk()

    def add_retrieved_docs_from_runfile(
        self,
        run_file_path: str,
        corpus: dict[str, Document] | dict[str, str],
        top_k: int | None = None,
    ):
        """
        Adds the retrieved documents from a run file to the queries.

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
        self,
        evaluation: (
            RetrievalEvaluatorResult | AnswerEvaluatorResult | EloTournamentResult
        ),
        should_save: bool = True,
    ):
        """
        Adds an evaluation to the queries and optionally saves the result.

        Args:
            evaluation (RetrievalEvaluatorResult | AnswerEvaluatorResult): The evaluation result to be added.
            should_save (bool): Flag indicating whether the result should be persist to disk. Defaults to True.
        Raises:
            ValueError: If the evaluation type is not supported.
        """
        if isinstance(evaluation, RetrievalEvaluatorResult):
            self._add_retrieval_evaluation(evaluation)
        elif isinstance(evaluation, AnswerEvaluatorResult):
            self._add_answer_evaluation(evaluation)
        elif isinstance(evaluation, EloTournamentResult):
            self.elo_tournaments.append(evaluation)
        else:
            raise ValueError(
                f"Cannot add evaluation of type {type(evaluation)} to queries"
            )
        if should_save:
            self.persist_result_to_cache(evaluation)

    def get_qrels(
        self,
        relevance_key: str | None = "answer",
        relevance_threshold: int = 0,
        output_path: str | None = None,
        output_format: str = "trec",
    ) -> dict[str, dict[str, int]]:
        """
        Retrieve the qrels (query relevance judgments) for the queries.
        Args:
            relevance_key (str | None): The key to use for determining relevance. Defaults to "answer".
            relevance_threshold (int): The threshold value for relevance. Defaults to 0.
        Returns:
            dict[str, dict[str, int]]: A dictionary where each key is a query ID and the value is another dictionary
                                       mapping document IDs to their relevance scores.
        """

        qrels = {}
        for qid, query in self.queries.items():
            qrels[qid] = query.get_qrels(relevance_key, relevance_threshold)
        if output_path:
            with open(output_path, "w") as f:
                if output_format.lower() == "trec":
                    for qid, qrel in qrels.items():
                        for did, rel in qrel.items():
                            f.write(f"{qid} Q0 {did} {rel}\n")
                elif output_format.lower() == "json":
                    json.dump(qrels, f)
                else:
                    raise ValueError(
                        f"Invalid output format for QRELS: {output_format}"
                        "Valid options are 'trec' and 'json'"
                    )
        return qrels

    def get_runs(
        self, agents: list[str] | None = None
    ) -> dict[str, dict[str, dict[str, float]]]:
        """
        Retrieve runs for specified agents.
        This method aggregates runs from all queries and organizes them by agent.
        Args:
            agents (list[str] | None): A list of agent names to filter the runs. If None, runs for all agents are retrieved.
        Returns:
            dict[str, dict[str, dict[str, float]]]: A nested dictionary where the first level keys are agent names,
            the second level keys are run identifiers, and the third level keys are metric names with their corresponding float values.
        """

        runs_by_agent = {}
        for query in self.queries.values():
            runs = query.get_runs(agents)
            for agent, runs in runs.items():
                runs_by_agent[agent] = runs_by_agent.get(agent, {}) | runs
        return runs_by_agent

    def evaluate_retrieval(
        self,
        metrics: list[str] = ["Precision@10", "nDCG@10", "Judged@10"],
        relevance_threshold: int = 0,
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate the retrieval performance of agents using specified metrics.
        Args:
            metrics (list[str]): A list of metric names to evaluate (default is ["Precision@10", "nDCG@10", "Judged@10"]).
        relevance_threshold (int): The threshold above which a document is considered relevant (default is 0).
        Returns:
            dict[str, dict[str, float]]: A dictionary where keys are agent names and values are dictionaries of metric scores.
        Raises:
            ImportError: If the `ir_measures` package is not installed.
            ValueError: If a specified metric is not found in the `ir_measures` registry.
        Notes:
        - The function uses the `ir_measures` package to calculate the metrics.
        - If the `rich` package is installed, the results are printed in a formatted table.
        - If the `rich` package is not installed, the results are printed using plain `print`.

        Example:
        >>> results = evaluate_retrieval(metrics=["Precision@10", "nDCG@10"], relevance_threshold=1)
        >>> print(results)
        {
            "agent1": {
                "Precision@10": 0.5,
                "nDCG@10": 0.6
            },
            "agent2": {
                "Precision@10": 0.4,
                "nDCG@10": 0.5
            }
        }
        """

        try:
            import ir_measures
            from ir_measures import parse_measure
        except ImportError:
            raise ImportError(
                "ir_measures is not installed. Please install it with `pip install ir-measures`"
            )
        qrels = self.get_qrels(relevance_threshold=relevance_threshold)
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

            rich.print(f"---[bold cyan] Retrieval Scores [/bold cyan] ---")
            if relevance_threshold > 0:
                rich.print(
                    f"[bold yellow]Relevance threshold: {relevance_threshold}[/bold yellow]"
                )
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

    def persist_on_disk(self):
        """
        Persist the current state of the model to disk if caching is enabled.
        This method checks if caching is enabled by evaluating the `save_cache` attribute.
        If caching is enabled, it attempts to write the model's current state to the file
        specified by `cache_path`. If `cache_path` is not set, a ValueError is raised.
        Raises:
            ValueError: If `cache_path` is None and caching is enabled.
        """

        if not self.save_cache:
            return
        if self.cache_path is None:
            raise ValueError("Cache path not set. Cannot dump queries")
        with open(self.cache_path, "w") as f:
            json.dump(self.model_dump(), f)

    def persist_result_to_cache(
        self,
        result: AnswerEvaluatorResult | RetrievalEvaluatorResult | EloTournamentResult,
    ):
        """
        Persist the evaluation result to a cache file.
        This method writes the given evaluation result to a cache file specified by
        `self.results_cache_path`. The result is serialized to JSON format and appended
        to the file. The type of the result (either "answer" or "retrieval") is included
        in the serialized data.
        Args:
            result (AnswerEvaluatorResult | RetrievalEvaluatorResult): The evaluation result
                to be persisted to the cache.
        Raises:
            ValueError: If `self.results_cache_path` is not set or if the type of `result`
                is not recognized.
        """

        if not self.save_cache:
            return
        if self.results_cache_path is None:
            raise ValueError("Results cache path not set. Cannot dump result")
        with open(self.results_cache_path, "a+") as f:
            if isinstance(result, AnswerEvaluatorResult):
                result_type = "answer"
            elif isinstance(result, RetrievalEvaluatorResult):
                result_type = "retrieval"
            elif isinstance(result, EloTournamentResult):
                result_type = "elo_tournament"
            else:
                raise ValueError(
                    f"Cannot save evaluation of type {type(result)} to cache"
                )
            f.write(json.dumps({result_type: result.model_dump()}) + "\n")

    def _add_retrieval_evaluation(self, evaluation: RetrievalEvaluatorResult):
        qid = evaluation.qid
        if qid not in self.queries:
            raise ValueError(
                f"Trying to add retrieval evaluation for non-existing query {qid}"
            )
        did = evaluation.did
        if did not in self.queries[qid].retrieved_docs:
            logger.warning(
                f"Trying to add evaluation for non-retrieved document {did} in query {qid}"
            )
            return
        if self.queries[qid].retrieved_docs[did].evaluation is not None:
            logger.info(
                f"Query {qid} already has an evaluation for document {did}. Overwriting."
            )
        self.queries[qid].retrieved_docs[did].evaluation = evaluation

    def _add_answer_evaluation(self, evaluation: AnswerEvaluatorResult):
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

                        logger.info(
                            f"Query {qid} already has an evaluation for agents {agent_a} and {agent_b}. Overwriting."
                        )
                    game.evaluation = evaluation
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

    def _clear_all_evaluations(self, should_save: bool = False):
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
        if should_save:
            if self.results_cache_path and os.path.isfile(self.results_cache_path):
                with open(self.results_cache_path, "w") as f:
                    pass
            self.persist_on_disk()

    def _read_queries_csv(self) -> dict[str, Query]:
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

    def _load_cached_queries(self):
        queries: dict[str, Query] = {}
        if not self.cache_path:
            raise ValueError("Cache path not set. Cannot load queries")
        if not os.path.isfile(self.cache_path):
            raise FileNotFoundError(f"Cache file {self.cache_path} not found")
        with open(self.cache_path) as f:
            queries = {k: Query(**v) for k, v in json.load(f)["queries"].items()}
        return queries

    def _load_results_from_cache(self):
        if not self.cache_path:
            raise ValueError("Cache path not set. Cannot load queries")
        if self.results_cache_path is None:
            self.results_cache_path = self.cache_path.replace(".json", "_results.jsonl")
            with open(self.results_cache_path, "w") as f:
                pass
        for line in open(self.results_cache_path):
            result = json.loads(line)
            result_type = list(result.keys())[0]
            if result_type == "answer":
                result = AnswerEvaluatorResult(**result["answer"])
                if result.qid not in self.queries:
                    continue
            elif result_type == "retrieval":
                result = RetrievalEvaluatorResult(**result["retrieval"])
                if result.qid not in self.queries:
                    continue
            elif result_type == "elo_tournament":
                result = EloTournamentResult(**result["elo_tournament"])
            self.add_evaluation(result, should_save=False)

    def __len__(self):
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries.values())

    def __getitem__(self, key: str) -> Query:
        return self.queries[key]

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
