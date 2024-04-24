"""Base model for dealing with answer evaluators"""

import asyncio
from abc import abstractmethod
from typing import Any, Callable, Optional, Type, get_type_hints

from tenacity import RetryError
from tqdm import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types import (
    AgentAnswer,
    AnswerEvaluatorResult,
    AnswerEvaluatorTypes,
    AnswerFormat,
    Document,
    Query,
)
from ragelo.types.configurations import BaseAnswerEvaluatorConfig
from ragelo.utils import load_retrieved_docs_from_csv


class BaseAnswerEvaluator(BaseEvaluator):
    config: BaseAnswerEvaluatorConfig
    output_columns = ["qid", "agent", "raw_answer", "answer"]
    output_file: str = "answers_evaluations.csv"

    def __init__(
        self,
        config: BaseAnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.config = config
        self.llm_provider = llm_provider
        if config.output_file is not None:
            self.output_file = config.output_file
        if config.answer_format == AnswerFormat.MULTI_FIELD_JSON:
            if isinstance(config.scoring_key, str):
                scoring_keys = [config.scoring_key]
            else:
                scoring_keys = config.scoring_key
            self.output_columns = ["qid", "agent", "raw_answer"] + scoring_keys

        if config.scoring_key and config.scoring_key not in self.output_columns:
            print(f"Adding scoring key {config.scoring_key} to output columns")
            self.output_columns.append(self.config.scoring_key)
        if config.scoring_keys:
            missing_keys = [
                key for key in config.scoring_keys if key not in self.output_columns
            ]
            self.output_columns.extend(missing_keys)

    def __get_tuples_to_evaluate(
        self, queries: list[Query], evaluations: list[AnswerEvaluatorResult]
    ) -> list[tuple[Query, AgentAnswer]]:
        skip_tuples = {(x.qid, x.agent) for x in evaluations}
        tuples_to_eval = []
        all_tuples = 0
        for query in queries:
            for agent_answer in query.answers:
                qid = query.qid
                agent = agent_answer.agent
                all_tuples += 1
                if (qid, agent) in skip_tuples:
                    logger.debug(f"Skipping {qid} {agent}")
                    continue
                tuples_to_eval.append((query, agent_answer))
        if len(tuples_to_eval) == 0:
            logger.info("All answers have been evaluated")
            if self.config.verbose:
                print(
                    f"All {all_tuples} answers are already evaluated.\n"
                    "If you want to re-evaluate them, use the force flag"
                )

        return tuples_to_eval

    async def __fetch_chunk(
        self, chunk: list[tuple[Query, AgentAnswer]]
    ) -> list[AnswerEvaluatorResult]:
        qids = []
        agent_ids = []
        tasks = []
        for query, agent_answer in chunk:
            message = self._build_message(query, agent_answer)
            qids.append(query.qid)
            agent_ids.append(agent_answer.agent)
            tasks.append(self.llm_provider.call_async(message))
        raw_answers = await asyncio.gather(*tasks)
        parsed_answers = []
        for qid, agent_id, raw_answer in zip(qids, agent_ids, raw_answers):
            try:
                answer = self._process_answer(raw_answer)
            except ValueError:
                logger.warning(
                    f"Failed to PARSE answer for qid: {qid} agent: {agent_id}"
                )
                continue
            parsed_answers.append(
                AnswerEvaluatorResult(
                    qid=qid,
                    agent=agent_id,
                    raw_answer=raw_answer,
                    answer=answer,
                )
            )
            self._dump_response(
                parsed_answers[-1], self.output_columns, self.output_file
            )
        return parsed_answers

    async def batch_evaluate_async(
        self, queries: list[Query]
    ) -> list[AnswerEvaluatorResult]:
        """Evaluate all the documents for a list of queries"""
        use_progress_bar = self.config.verbose
        answers = [AnswerEvaluatorResult(**x) for x in self._get_existing_output()]
        queries = self._add_retrieved_documents_to_queries(
            queries, self.config.documents_path
        )

        tuples_to_eval = self.__get_tuples_to_evaluate(queries, answers)
        if len(tuples_to_eval) == 0:
            return answers

        chunks = [
            tuples_to_eval[i : i + self.config.n_processes]
            for i in range(0, len(tuples_to_eval), self.config.n_processes)
        ]
        pbar = tqdm(
            total=len(tuples_to_eval),
            ncols=100,
            desc="Evaluating documents",
            disable=not use_progress_bar,
            leave=False,
            position=0,
        )

        for chunk in chunks:
            responses = await self.__fetch_chunk(chunk)
            answers.extend(responses)
            pbar.update(len(chunk))
        pbar.close()

        if self.config.verbose:
            print("✅ Done!")
            print(f"Total evaluations: {len(answers)}")

        return answers

    def batch_evaluate(self, queries: list[Query]) -> list[AnswerEvaluatorResult]:
        use_progress_bar = self.config.verbose
        failed_evaluations = 0
        evaluations = [AnswerEvaluatorResult(**x) for x in self._get_existing_output()]
        queries = self._add_retrieved_documents_to_queries(
            queries, self.config.documents_path
        )

        tuples_to_eval = self.__get_tuples_to_evaluate(queries, evaluations)
        if len(tuples_to_eval) == 0:
            return evaluations

        for query, agent_answer in tqdm(
            tuples_to_eval,
            desc="Annotating Answers",
            disable=not use_progress_bar,
            ncols=100,
            leave=False,
            position=0,
        ):
            qid = query.qid
            agent = agent_answer.agent
            try:
                raw_answer, answer = self.evaluate(query, agent_answer)
            except (RetryError, ValueError):
                failed_evaluations += 1
                continue
            evaluations.append(
                AnswerEvaluatorResult(
                    qid=qid, agent=agent, raw_answer=raw_answer, answer=answer
                )
            )
            self._dump_response(evaluations[-1], self.output_columns, self.output_file)
        if self.config.verbose:
            print("✅ Done!")
            print(f"Unparsed answers: {failed_evaluations}")
            print(f"Total evaluations: {len(evaluations)}")
        return evaluations

    def evaluate(
        self,
        query: Query | str,
        answer: AgentAnswer | str,
        retrieved_documents: Optional[list[str] | list[Document]] = None,
        document_metadata: Optional[list[dict[str, Any]]] = None,
        query_metadata: Optional[dict[str, Any]] = None,
        answer_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Any]:
        query = self._assemble_query(query, query_metadata)
        answer = self._assemble_answer(answer, answer_metadata)
        if isinstance(retrieved_documents, str):
            retrieved_documents = [retrieved_documents]
        if retrieved_documents:
            retrieved_and_assembled_docs = self._assemble_documents(
                retrieved_documents, document_metadata
            )
            query.retrieved_docs = retrieved_and_assembled_docs

        message = self._build_message(query, answer)
        try:
            raw_answer = self.llm_provider(message)
        except RetryError as e:
            logger.warning(
                f"Failed to fetch answer for qid: {query.qid} agent: {answer.agent}"
            )
            raise e
        try:
            processed_answer = self._process_answer(raw_answer)

        except ValueError as e:
            logger.warning(
                f"Failed to parse answer for qid: {query.qid} agent: {answer.agent}"
                f"Full answer: {raw_answer}"
            )
            raise e
        return raw_answer, processed_answer

    @abstractmethod
    def _build_message(
        self, query: Query, answer: AgentAnswer
    ) -> str | list[dict[str, str]]:
        """Builds the message to send to the LLM evaluator"""
        raise NotImplementedError

    @classmethod
    def from_config(
        cls, config: BaseAnswerEvaluatorConfig, llm_provider: BaseLLMProvider
    ):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> Type[BaseAnswerEvaluatorConfig]:
        return get_type_hints(cls)["config"]

    @staticmethod
    def _construct_list_of_answers(
        answers: list[dict[str, str]]
    ) -> list[AnswerEvaluatorResult]:
        return [AnswerEvaluatorResult(**x) for x in answers]

    @staticmethod
    def _prepare_documents(query: Query) -> str:
        documents = [(d.did, d.text) for d in query.retrieved_docs]
        if len(documents) == 0:
            return "NO DOCUMENTS WERE RETRIEVED"
        return "\n".join([f"[{did}] {doc.strip()}" for did, doc in documents])

    def _add_retrieved_documents_to_queries(
        self,
        queries: list[Query],
        documents_path: Optional[str],
        text_column: str = "document_text",
        overwrite: bool = False,
    ):
        if all([len(q.retrieved_docs) > 0 for q in queries]) and not overwrite:
            logger.info("All queries already have retrieved documents")
            return queries
        if documents_path is None:
            logger.warning(
                "No path with retrieved documents provided."
                "Evaluator performance may be affected."
            )
            return queries
        queries_with_docs = load_retrieved_docs_from_csv(
            documents_path, queries, document_text_col=text_column
        )
        return queries_with_docs


class AnswerEvaluatorFactory:
    registry: dict[AnswerEvaluatorTypes | str, Type[BaseAnswerEvaluator]] = {}

    @classmethod
    def register(cls, name: AnswerEvaluatorTypes) -> Callable:
        def inner_wrapper(wrapped_class: Type[BaseAnswerEvaluator]):
            if name in cls.registry:
                logger.warning(f"Overwriting {name} in registry")
            cls.registry[name.lower()] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(
        cls,
        evaluator_name: str,
        llm_provider: BaseLLMProvider | str,
        config: Optional[BaseAnswerEvaluatorConfig] = None,
        **kwargs,
    ) -> BaseAnswerEvaluator:
        if evaluator_name.lower() not in cls.registry:
            raise ValueError(f"Unknown evaluator {evaluator_name}")
        if isinstance(llm_provider, str):
            llm_provider_instance = get_llm_provider(llm_provider, **kwargs)
        else:
            llm_provider_instance = llm_provider
        if config is None:
            class_ = cls.registry[evaluator_name]
            type_config = class_.get_config_class()
            valid_keys = [field for field in type_config.get_model_fields()]
            valid_args = {k: v for k, v in kwargs.items() if k in valid_keys}
            config = type_config(**valid_args)
        return cls.registry[evaluator_name.lower()].from_config(
            config, llm_provider_instance
        )


def get_answer_evaluator(
    evaluator_name: AnswerEvaluatorTypes | str,
    llm_provider: BaseLLMProvider | str,
    config: Optional[BaseAnswerEvaluatorConfig] = None,
    **kwargs,
) -> BaseAnswerEvaluator:
    return AnswerEvaluatorFactory.create(
        evaluator_name,
        llm_provider=llm_provider,
        config=config,
        **kwargs,
    )
