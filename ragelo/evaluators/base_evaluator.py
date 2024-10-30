from __future__ import annotations

import asyncio
import json
import string
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from tqdm.auto import tqdm

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
from ragelo.types.configurations import BaseEvaluatorConfig
from ragelo.types.evaluables import AgentAnswer, Document, Evaluable
from ragelo.types.formats import AnswerFormat
from ragelo.types.queries import Queries
from ragelo.types.query import Query
from ragelo.types.results import EvaluatorResult


class BaseEvaluator(ABC):
    config: BaseEvaluatorConfig
    scoring_keys: list[str]
    scoring_key: str
    answer_format: AnswerFormat

    @abstractmethod
    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        config: BaseEvaluatorConfig,
    ):
        raise NotImplementedError

    async def _async_batch_evaluate(self, queries: Queries):
        tuples_to_eval = self._get_tuples_to_evaluate(queries)
        if self.config.rich_print:
            import warnings

            from tqdm import TqdmExperimentalWarning

            warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
            from tqdm.rich import tqdm as rich_tqdm

            pbar_fn = rich_tqdm  # type: ignore
        else:
            pbar_fn = tqdm  # type: ignore

        pbar = pbar_fn(
            total=len(tuples_to_eval),
            ncols=100,
            desc="Evaluating retrieved documents",
            disable=not self.config.use_progress_bar,
            leave=False,
            position=0,
        )
        awaitables_ended = False
        pending: set[asyncio.Future] = set()
        aws = map(self.evaluate_async, tuples_to_eval)
        aws = iter(aws)
        failed = 0
        evaluations = 0
        while pending or not awaitables_ended:
            while len(pending) < self.config.n_processes and not awaitables_ended:
                try:
                    aw = next(aws)
                except StopIteration:
                    awaitables_ended = True
                else:
                    pending.add(asyncio.ensure_future(aw))
            if not pending:
                break
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            while done:
                evaluation = await done.pop()
                evaluations += 1
                pbar.update()
                if evaluation.exception:
                    failed += 1
                    continue
                queries.add_evaluation(evaluation)
        pbar.close()
        if self.config.verbose:
            self._print_failed_evaluations(evaluations, failed)

    def batch_evaluate(self, queries: Queries):
        def run(coroutine):
            return asyncio.run(coroutine)

        try:
            asyncio.get_running_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run, self._async_batch_evaluate(queries))
                _ = future.result()
        except RuntimeError:
            _ = asyncio.run(self._async_batch_evaluate(queries))

    @abstractmethod
    def _get_tuples_to_evaluate(
        self, queries: Queries
    ) -> list[tuple[Query, Evaluable]]:
        raise NotImplementedError

    @abstractmethod
    async def evaluate_async(
        self,
        eval_sample: tuple[Query, Evaluable],
    ) -> EvaluatorResult:
        raise NotImplementedError

    def _validate_answer(
        self,
        answer: dict[str, Any] | PydanticBaseModel | str,
    ):
        """Ensures that the LLM output is properly formatted."""

        if self.config.llm_answer_format == AnswerFormat.JSON:
            if not isinstance(answer, dict):
                raise ValueError(
                    f"Expected LLM answer as a JSON dictionary, got {type(answer)}: {answer}"
                )
            if self.config.llm_response_schema is not None:
                if isinstance(self.config.llm_response_schema, dict):
                    schema = self.config.llm_response_schema
                    if not all(k in answer for k in schema.keys()):
                        raise ValueError(
                            f"Expected LLM answer to have keys {schema.keys()}, got {answer.keys()}"
                        )
        elif self.config.llm_answer_format == AnswerFormat.STRUCTURED:
            if not isinstance(answer, PydanticBaseModel):
                raise ValueError(
                    f"Expected LLM answer as a PydanticBaseModel, got {type(answer)}: {answer}"
                )
        elif self.config.llm_answer_format == AnswerFormat.TEXT:
            if not isinstance(answer, str):
                raise ValueError(
                    f"Expected LLM answer as a string, got {type(answer)}: {answer}"
                )

    def _process_answer(
        self, raw_answer: str | dict[str, Any] | PydanticBaseModel
    ) -> int | str | dict[str, Any] | PydanticBaseModel:
        """Processes the raw answer returned by the LLM. Should be implemented by the subclass if needed."""
        self._validate_answer(raw_answer)
        return raw_answer

    @staticmethod
    def _get_fields_from_string(s: str) -> list[str]:
        """Parse a formatted string and return all the fields in it"""
        field_names = [v[1] for v in string.Formatter().parse(s) if v[1] is not None]
        return field_names

    @staticmethod
    def _get_usable_fields_from_metadata(
        prompt: str, metadata: dict[str, str] | None, skip_fields: list[str] = []
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
    def _print_response(
        response: EvaluatorResult,
        rich_print: bool = False,
    ):
        answer: str | dict[str, str] | int | None
        if isinstance(response.answer, dict):
            # Print the answer in a more readable format
            answer = json.dumps(response.answer, indent=4)
        elif isinstance(response.answer, PydanticBaseModel):
            answer = response.answer.model_dump_json(indent=4)
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
                    rich.print(f"[bold bright_cyan]🕵️ Agent[/bold bright_cyan]: {agent}")
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
    def _assemble_query(
        query: Query | str, query_metadata: dict[str, Any] | None = None
    ) -> Query:
        if isinstance(query, str):
            query = Query(qid="<no_qid>", query=query)
        query.add_metadata(query_metadata)
        return query

    @staticmethod
    def _assemble_document(
        document: Document | str, doc_metadata: dict[str, Any] | None = None
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
        doc_metadata: list[dict[str, Any]] | list[None] | None = None,
    ) -> dict[str, Document]:
        assembled_docs: dict[str, Document] = {}
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
        answer: AgentAnswer | str,
        answer_metadata: dict[str, Any] | None = None,
    ) -> AgentAnswer:
        if isinstance(answer, str):
            answer = AgentAnswer(agent="<no_agent>", text=answer)
        answer.add_metadata(answer_metadata)
        return answer

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
