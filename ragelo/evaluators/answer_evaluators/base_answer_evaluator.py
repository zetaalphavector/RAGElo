"""Base model for dealing with answer evaluators"""

import csv
import os
import dataclasses
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Optional, Type, get_type_hints

from tenacity import RetryError
from tqdm import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types import AgentAnswer, AnswerEvaluatorResult, Query, AnswerEvaluatorTypes
from ragelo.types.configurations import BaseAnswerEvaluatorConfig


class BaseAnswerEvaluator(BaseEvaluator):
    config: BaseAnswerEvaluatorConfig
    output_columns = ["qid", "agent", "raw_answer", "answer"]
    output_file: str = "answers_evaluations.csv"
    tuple_columns: list[str] = ["qid", "agent"]

    def __init__(
        self,
        config: BaseAnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        self.config = config
        self.llm_provider = llm_provider
        if config.output_file is not None:
            self.output_file = config.output_file
        self.reasonings = self._load_reasonings(self.config.reasoning_path)

    def batch_evaluate(self, queries: list[Query]) -> list[AnswerEvaluatorResult]:
        use_progress_bar = self.config.verbose
        failed_evaluations = 0
        skip_tuples = self._get_skip_tuples()
        evaluations: list[AnswerEvaluatorResult] = []
        for query in tqdm(
            queries,
            desc="Annotating Answers",
            disable=not use_progress_bar,
            ncols=100,
            leave=False,
            position=0,
        ):
            for agent_answer in tqdm(
                query.answers,
                desc=query.qid,
                disable=not use_progress_bar,
                ncols=100,
                leave=False,
                position=1,
            ):
                qid = query.qid
                agent = agent_answer.agent
                if (qid, agent) in skip_tuples:
                    logger.debug(f"Skipping {qid} {agent}")
                    continue
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
                self._dump_response(
                    evaluations[-1], self.output_columns, self.output_file
                )
        if self.config.verbose:
            print("✅ Done!")
            print(f"Unparsed answers: {failed_evaluations}")
            print(f"Total evaluations: {len(evaluations)}")
        return evaluations

    def evaluate(
        self,
        query: Query | str,
        answer: AgentAnswer | str,
        query_metadata: Optional[dict[str, Any]] = None,
        answer_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, Any]:
        if isinstance(query, str):
            query = Query(qid="<no_qid>", query=query)
        if isinstance(answer, str):
            answer = AgentAnswer(agent="<no_agent>", text=answer)
        if query_metadata:
            if query.metadata is not None:
                logger.warning(
                    f"Query metadata for query id {query.qid} is being overwritten!\n"
                    f"Old metadata: {query.metadata}\n"
                    f"New metadata: {query_metadata}\n"
                )
            query.metadata = query_metadata
        if answer_metadata:
            if answer.metadata is not None:
                logger.warning(
                    f"Answer metadata for query id {query.qid} and agent {answer.agent} is being overwritten!\n"
                    f"Old metadata: {answer.metadata}\n"
                    f"New metadata: {answer_metadata}\n"
                )
            answer.metadata = answer_metadata

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
        self, query: Query, answer: AgentAnswer | tuple[AgentAnswer, AgentAnswer]
    ) -> str | list[dict[str, str]]:
        """Builds the message to send to the LLM evaluator"""
        raise NotImplementedError

    @abstractmethod
    def _process_answer(self, answer: str) -> Any:
        """Processes the LLM evaluator output into some serializable format"""
        raise NotImplementedError

    def _load_reasonings(
        self,
        reasoning_path: str,
        query_id_col: str = "qid",
        document_id_col: str = "did",
        answer_col: str = "answer",
    ) -> dict[str, dict[str, str]]:
        reasoning: dict[str, dict[str, str]] = defaultdict(lambda: dict())
        reasoning_read = 0
        if not os.path.exists(reasoning_path):
            raise FileNotFoundError(f"Reasoning file {reasoning_path} not found")
        for line in csv.DictReader(open(reasoning_path)):
            reasoning_read += 1
            reasoning[line[query_id_col]][line[document_id_col]] = line[answer_col]
        logger.info(f"Loaded {reasoning_read} reasonings")
        return dict(reasoning)

    def _prepare_reasonings(self, qid: str) -> str:
        return "\n".join(
            [" ".join([f"[{idx}]", r]) for (idx, r) in self.reasonings[qid].items()]
        )

    @classmethod
    def from_config(
        cls, config: BaseAnswerEvaluatorConfig, llm_provider: BaseLLMProvider
    ):
        return cls(config, llm_provider)

    @classmethod
    def get_config_class(cls) -> Type[BaseAnswerEvaluatorConfig]:
        return get_type_hints(cls)["config"]


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
            valid_keys = [field.name for field in dataclasses.fields(type_config)]
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
