import csv
import json
import os
import string
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Union

from tqdm.auto import tqdm

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
from ragelo.types import (
    AgentAnswer,
    AnswerEvaluatorResult,
    Document,
    EvaluatorResult,
    Query,
    RetrievalEvaluatorResult,
)
from ragelo.types.configurations import AnswerFormat, BaseEvaluatorConfig


class BaseEvaluator(ABC):
    config: BaseEvaluatorConfig
    scoring_keys: List[str]
    scoring_key: str
    answer_format: AnswerFormat

    @abstractmethod
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        config: BaseEvaluatorConfig,
    ):
        raise NotImplementedError

    @staticmethod
    def __parse_json(answer: str, keys: Union[str, List[str]]) -> Dict[str, Any]:
        # Checks if there is block of json in the answer like ```json\n{...}\n```
        json_block = answer.split("```json\n")
        if len(json_block) > 1:
            json_data = json_block[1].split("\n```")[0]
            return json.loads(json_data)
        # Otherwise, go line by line
        if isinstance(keys, str):
            keys = [keys]
        json_dict = {}
        for line in answer.strip().split("\n"):
            try:
                json_object = json.loads(line)
                for k in keys:
                    if k in json_object:
                        json_dict[k] = json_object[k]
            except json.JSONDecodeError:
                pass
        return json_dict

    def json_answer_parser(self, answer: str, key: str) -> str:
        """Parses a Json answer from the LLM and returns a specific key"""

        # Finds all valid JSON objects in the answer that contain the key
        json_dict = self.__parse_json(answer, key)
        if key not in json_dict:
            raise ValueError(
                "Answer does not contain the necessary key\n"
                f"Expected {key}, found {json_dict.keys()}\n{answer}"
            )
        return json_dict[key]

    def json_answer_parser_multifields(
        self, answer: str, keys: List[str]
    ) -> Dict[str, str]:
        """Parses a Json answer from the LLM and returns the values from multiple fields"""
        # Finds all valid JSON objects in the answer that contain the key
        try:
            json_dict = json.loads(answer)
        except json.JSONDecodeError:
            json_dict = self.__parse_json(answer, keys)

        valid_keys = set(json_dict.keys()).intersection(keys)
        if len(valid_keys) != len(keys):
            raise ValueError(
                "Answer does not contain all necessary keys\n"
                f"Expected {keys}, found {valid_keys}.\n"
                f"Full Answer:\n{answer}"
            )
        return json_dict

    @staticmethod
    def _get_fields_from_string(s: str) -> List[str]:
        """Parse a formatted string and return all the fields in it"""
        field_names = [v[1] for v in string.Formatter().parse(s) if v[1] is not None]
        return field_names

    @staticmethod
    def _get_usable_fields_from_metadata(
        prompt: str, metadata: Optional[Dict[str, str]], skip_fields: List[str] = []
    ) -> Dict[str, str]:
        """Get the fields from the prompt that are in the metadata"""
        expected_fields = BaseEvaluator._get_fields_from_string(prompt)
        valid_fields: Dict[str, str] = {}
        if metadata is None:
            return valid_fields
        for field in expected_fields:
            if field in metadata and field not in skip_fields:
                valid_fields[field] = metadata[field]
        return valid_fields

    @staticmethod
    def __get_existing_evaluations_from_json(output_file: str) -> List[Dict[str, str]]:
        return json.load(open(output_file, "r"))

    @staticmethod
    def __get_existing_evaluations_from_jsonl(output_file: str) -> List[Dict[str, str]]:
        with open(output_file, "r") as f:
            return [json.loads(line) for line in f]

    @staticmethod
    def __get_existing_evaluations_from_csv(output_file: str) -> List[Dict[str, Any]]:
        existing_lines: List[Dict[str, str]] = []
        base_columns = ["qid", "did", "agent", "raw_answer", "agent_a", "agent_b"]
        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            line: Dict[str, Any]
            for line in reader:
                line_dict = line
                if "answer" in line and line["answer"]:
                    line_dict["answer"] = line["answer"]
                else:
                    remaining_keys = {k for k in line.keys() if k not in base_columns}
                    line_dict["answer"] = {k: line[k] for k in remaining_keys}
                existing_lines.append(line_dict)
        return existing_lines

    def _get_existing_evaluations(
        self, evaluation_file: str, force: bool = False
    ) -> List[Dict[str, Any]]:
        existing_lines: List[Dict[str, str]] = []
        if force and os.path.exists(evaluation_file):
            logger.warning(f"Removing existing {evaluation_file}!")
            os.remove(evaluation_file)
        if not os.path.isfile(evaluation_file):
            return existing_lines
        if evaluation_file.endswith(".json"):
            return self.__get_existing_evaluations_from_json(evaluation_file)
        if evaluation_file.endswith(".jsonl"):
            return self.__get_existing_evaluations_from_jsonl(evaluation_file)
        else:
            return self.__get_existing_evaluations_from_csv(evaluation_file)

    @staticmethod
    def _print_response(
        response: EvaluatorResult,
        rich_print: bool = False,
    ):
        answer: Optional[Union[str, Dict[str, str], int]]
        if isinstance(response.answer, dict):
            # Print the answer in a more readable format
            answer = json.dumps(response.answer, indent=4)
        else:
            answer = response.answer
        response_dict = response.model_dump()
        agent_a = response_dict.get("agent_a")
        agent_b = response_dict.get("agent_b")
        agent = response_dict.get("agent")
        qid = response_dict.get("qid")
        did = response_dict.get("did")
        raw_answer = response_dict.get("raw_answer")

        if rich_print:
            try:
                import rich

                rich.print(f"[bold blue]🔎 Query ID[/bold blue]: {qid}")
                did = response_dict.get("did")
                if did:
                    rich.print(f"[bold blue]📜 Document ID[/bold blue]: {did}")
                if agent_a and agent_b:
                    rich.print(
                        f"[bold bright_cyan] {agent_a:<18} [/bold bright_cyan] 🆚  "
                        f"[bold red] {agent_b}[/bold red]"
                    )
                elif agent:
                    rich.print(
                        f"[bold bright_cyan]🕵️ Agent[/bold bright_cyan]: {agent}"
                    )
                if raw_answer != answer:
                    rich.print(f"[bold blue]Raw Answer[/bold blue]: {raw_answer}")
                rich.print(f"[bold blue]Parsed Answer[/bold blue]: {answer}")
                rich.print("")
                return
            except ImportError:
                logger.warning("Rich not installed. Using plain print")
        tqdm.write(f"Query ID: {qid}")
        if did:
            tqdm.write(f"Document ID: {did}")
        if agent_a and agent_b:
            tqdm.write(f"{agent_a} vs {agent_b}")
        elif agent:
            tqdm.write(f"Agent: {agent}")
        tqdm.write(f"Raw Answer: {raw_answer}")
        tqdm.write(f"Parsed Answer: {answer}")
        tqdm.write("")

    @staticmethod
    def __dump_response_csv(
        answer_dict: Dict[str, Any], output_columns: List[str], output_file: str
    ):
        if not any(k in output_columns for k in answer_dict.keys()):
            raise ValueError(
                "No parsed answer fields are in the output columns. \n"
                f"Expected output columns: {output_columns}. \n"
                f"Answer fields: {answer_dict.keys()}"
            )
        answer_dict = {k: answer_dict.get(k, None) for k in output_columns}
        if not os.path.isfile(output_file):
            logger.debug(f"Creating new file {output_file}")
            with open(output_file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=output_columns)
                writer.writeheader()
        with open(output_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=output_columns)
            writer.writerow(answer_dict)

    @staticmethod
    def __dump_response_jsonl(answer_dict: Dict[str, str], output_file: str):
        with open(output_file, "a") as f:
            f.write(json.dumps(answer_dict) + "\n")

    @staticmethod
    def __dump_response_json(answer_dict: Dict[str, str], output_file: str):
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
        response: EvaluatorResult,
        output_columns: List[str],
        output_file: str,
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
        if self.answer_format == AnswerFormat.JSON:
            return self.json_answer_parser(answer, self.scoring_key)
        if self.answer_format == AnswerFormat.MULTI_FIELD_JSON:
            return self.json_answer_parser_multifields(answer, self.scoring_keys)
        if self.answer_format == AnswerFormat.TEXT:
            return answer

    @staticmethod
    def _assemble_query(
        query: Union[Query, str], query_metadata: Optional[Dict[str, Any]] = None
    ) -> Query:
        if isinstance(query, str):
            query = Query(qid="<no_qid>", query=query)
        query.add_metadata(query_metadata)
        return query

    @staticmethod
    def _assemble_document(
        document: Union[Document, str], doc_metadata: Optional[Dict[str, Any]] = None
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
        documents: Union[List[str], List[Document]],
        doc_metadata: Optional[List] = None,
    ) -> Dict[str, Document]:
        assembled_docs: Dict[str, Document] = {}
        if doc_metadata and len(documents) != len(doc_metadata):
            raise ValueError(
                "The number of documents and document metadata do not match"
            )
        if not doc_metadata:
            doc_metadata = [None] * len(documents)

        for idx, (doc, m) in enumerate(zip(documents, doc_metadata)):
            if isinstance(doc, str):
                doc = Document(did=f"doc_{idx}", text=doc, metadata=m)
            else:
                doc.add_metadata(m)  # type: ignore
            assembled_docs[doc.did] = doc  # type: ignore
        return assembled_docs

    @staticmethod
    def _assemble_answer(
        answer: Union[AgentAnswer, str],
        answer_metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentAnswer:
        if isinstance(answer, str):
            answer = AgentAnswer(agent="<no_agent>", text=answer)
        answer.add_metadata(answer_metadata)
        return answer

    def _load_retrieved_documents(
        self,
        queries: List[Query],
        query_id_col: str = "qid",
        document_id_col: str = "did",
        document_text_col: str = "document_text",
    ) -> List[Query]:
        # Check if we actually need to do something
        queries_with_documents = len([q for q in queries if len(q.retrieved_docs) > 0])
        if queries_with_documents == len(queries):
            logger.debug("All Queries already have retrieved documents. Skipping")
            return queries
        documents_read = 0
        if not os.path.isfile(self.config.documents_file):
            logger.warning(
                f"Documents file {self.config.documents_file} not found"
                "Will not add retrieved documents to queries"
                f"{queries_with_documents} queries already have documents"
            )
            return queries

        queries_idx = {q.qid: idx for idx, q in enumerate(queries)}
        docs_per_query: Dict[str, Set[str]] = {}
        for line in csv.DictReader(open(self.config.documents_file)):
            qid = line[query_id_col].strip()
            did = line[document_id_col].strip()
            text = line[document_text_col].strip()
            extra_metadata = {
                k: v
                for k, v in line.items()
                if k not in [query_id_col, document_id_col, document_text_col]
            }
            if qid not in queries_idx:
                logger.info(f"Query {qid} not in the provided queries. Skipping")
                continue
            if qid not in docs_per_query:
                docs_per_query[qid] = set()
            if did in docs_per_query[qid]:
                logger.info(
                    f"Document {did} already in the retrieved documents for query {qid}. Skipping"
                )
                continue
            docs_per_query[qid].add(did)
            queries[queries_idx[qid]].add_retrieved_doc(
                Document(did=did, text=text, metadata=extra_metadata or None)
            )
            documents_read += 1
        logger.info(f"Loaded {documents_read} documents")
        return queries

    def _load_agent_answers(
        self,
        queries: List[Query],
        query_id_col: str = "qid",
        agent_col: str = "agent",
        answer_col: str = "answer",
    ):
        queries_with_answers = len([q for q in queries if len(q.answers) > 0])
        if queries_with_answers == len(queries):
            logger.debug("All Queries already have answers. Skipping")
            return queries
        if not os.path.isfile(self.config.answers_file):
            logger.warning(
                f"Answers file {self.config.answers_file} not found. Will not add answers to queries"
                f"Queries with answers: {queries_with_answers}"
            )
            return queries
        queries_idx = {q.qid: idx for idx, q in enumerate(queries)}
        answers_read = 0
        answers_per_query: Dict[str, Set[str]] = {}
        for line in csv.DictReader(open(self.config.answers_file)):
            qid = line[query_id_col].strip()
            agent = line[agent_col].strip()
            answer = line[answer_col].strip()
            extra_metadata = {
                k: v
                for k, v in line.items()
                if k not in [query_id_col, agent_col, answer_col]
            }
            if qid not in queries_idx:
                logger.info(f"Query {qid} not in the provided queries. Skipping")
                continue
            if qid not in answers_per_query:
                answers_per_query[qid] = set()
            if agent in answers_per_query[qid]:
                logger.info(
                    f"Answer for agent {agent} already in the answers for query {qid}. Skipping"
                )
                continue
            answers_per_query[qid].add(agent)
            ans = AgentAnswer(agent=agent, text=answer, metadata=extra_metadata or None)
            queries[queries_idx[qid]].add_agent_answer(ans)
            answers_read += 1
        logger.info(f"Loaded {answers_read} answers")
        return queries

    def _load_document_evaluations(
        self, queries: List[Query], force: bool = False
    ) -> List[Query]:
        if force:
            logger.info("Clearing existing document evaluations")
            for q in queries:
                for doc in q.retrieved_docs.values():
                    doc.evaluation = None
        document_evaluations = [
            RetrievalEvaluatorResult(**x)
            for x in self._get_existing_evaluations(
                self.config.document_evaluations_file,
                force,
            )
        ]
        return self._add_document_evaluations(queries, document_evaluations)

    def _add_document_evaluations(
        self,
        queries: List[Query],
        evaluations: List[RetrievalEvaluatorResult],
    ) -> List[Query]:
        queries_idx = {q.qid: idx for idx, q in enumerate(queries)}

        for evaluation in evaluations:
            qid = evaluation.qid
            did = evaluation.did
            if qid not in queries_idx:
                logger.info(f"Query {qid} not in the provided queries. Skipping")
                continue
            if did not in queries[queries_idx[qid]].retrieved_docs:
                logger.info(
                    f"Document {did} not in the retrieved documents for query {qid}. Skipping"
                )
                continue
            queries[queries_idx[qid]].retrieved_docs[did].evaluation = evaluation
        return queries

    def _load_answers_evaluations(
        self, queries: List[Query], force: bool = False
    ) -> List[Query]:
        if force:
            logger.info("Clearing existing answers evaluations")
            for q in queries:
                for ans in q.answers.values():
                    ans.evaluation = None
                for game in q.pairwise_games:
                    game.evaluation = None
        evaluations = [
            AnswerEvaluatorResult(**x)
            for x in self._get_existing_evaluations(
                self.config.answers_evaluations_file,
                force,
            )
        ]
        evaluations += [
            AnswerEvaluatorResult(**x)
            for x in self._get_existing_evaluations(
                self.config.games_evaluations_file, force
            )
        ]
        return self._add_answers_evaluations(queries, evaluations)

    def _add_answers_evaluations(
        self, queries: List[Query], evaluations: List[AnswerEvaluatorResult]
    ) -> List[Query]:
        queries_idx = {q.qid: idx for idx, q in enumerate(queries)}
        games_idxs = {}
        for q in queries:
            games_idxs[q.qid] = {
                (g.agent_a_answer.agent, g.agent_b_answer.agent): idx
                for idx, g in enumerate(q.pairwise_games)
            }

        for evaluation in evaluations:
            query_idx = queries_idx[evaluation.qid]
            if evaluation.pairwise:
                if not evaluation.agent_a or not evaluation.agent_b:
                    # Should never happen, as the pydantic model enforces this
                    raise ValueError("Pairwise evaluations require two agents")
                agents = (evaluation.agent_a, evaluation.agent_b)
                if agents not in games_idxs[evaluation.qid]:
                    agents = (evaluation.agent_b, evaluation.agent_a)
                    if agents not in games_idxs[evaluation.qid]:
                        logger.info(
                            f"Pairwise evaluation between {evaluation.agent_a} and {evaluation.agent_b} "
                            f"not found in query {evaluation.qid}"
                        )
                        continue

                game_idx = games_idxs[evaluation.qid][agents]
                queries[query_idx].pairwise_games[game_idx].evaluation = evaluation
            else:
                if evaluation.agent is None:
                    # Should never happen.
                    raise ValueError("Evaluation must have an agent")
                queries[query_idx].answers[evaluation.agent].evaluation = evaluation
        return queries

    def _print_failed_evaluations(
        self, total_evaluations: int, failed_evaluations: int
    ):
        if self.config.rich_print:
            try:
                import rich

                rich.print("✅ Done!")
                rich.print(f"Failed evaluations: {failed_evaluations}")
                rich.print(f"Total evaluations: {total_evaluations}")
                return
            except ImportError:
                logger.warning("Rich not installed. Using plain print")
        print("✅ Done!")
        print(f"Failed evaluations: {failed_evaluations}")
        print(f"Total evaluations: {total_evaluations}")
