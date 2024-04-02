"""Base model for dealing with answer evaluators"""

import csv
import dataclasses
import logging
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Callable, Optional, Sequence, Type, get_type_hints

from tenacity import RetryError
from tqdm import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider
from ragelo.logger import logger
from ragelo.types import AgentAnswer, AnswerEvaluatorTypes
from ragelo.types.configurations import BaseAnswerEvaluatorConfig


class BaseAnswerEvaluator(BaseEvaluator):
    output_columns = ["query_id", "agent", "raw_answer", "answer"]
    config: BaseAnswerEvaluatorConfig
    output_file: str = "answers_evaluations.csv"
    tuple_columns: list[str] = ["query_id", "agent"]

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

    @classmethod
    def from_config(
        cls, config: BaseAnswerEvaluatorConfig, llm_provider: BaseLLMProvider
    ):
        return cls(config, llm_provider)

    def run(self, answers: dict[str, list[AgentAnswer]]) -> list[dict[str, str]]:
        use_progress_bar = self.config.verbose
        unparsed_answers = 0
        skip_tuples = self._get_skip_tuples()
        evaluations: list[dict[str, str]] = []
        for qid in tqdm(
            answers,
            desc="Annotating Answers",
            disable=not use_progress_bar,
            ncols=100,
            leave=False,
        ):
            for answer in tqdm(
                answers[qid],
                desc=qid,
                disable=not use_progress_bar,
                ncols=100,
                leave=False,
            ):
                if (qid, answer.agent) in skip_tuples:
                    logging.debug(f"Skipping {qid} {answer.agent}")
                    continue
                try:
                    answer_dict = self.evaluate_single_sample(answer)
                except RetryError:
                    continue
                except ValueError:
                    unparsed_answers += 1
                    continue
                self._dump_response(answer_dict)
                evaluations.append(answer_dict)
        if self.config.verbose:
            print("✅ Done!")
            print(f"Unparsed answers: {unparsed_answers}")
            print(f"Total evaluations: {len(evaluations)}")
        return evaluations

    def evaluate_single_sample(self, answer) -> dict[str, str]:
        message = self._build_message(answer)
        try:
            raw_answer = self.llm_provider(message)
        except RetryError as e:
            logger.warning(
                f"Failed to get answer for {answer.query.qid} {answer.agent}"
            )
            raise e
        try:
            processed_answer = self._process_answer(raw_answer)

        except ValueError as e:
            logger.error(
                f"Failed to parse answer for {answer.query.qid} {answer.agent}"
                f"Full answer: {raw_answer}"
            )
            raise e
        if isinstance(processed_answer, dict):
            return {
                "query_id": answer.query.qid,
                "agent": answer.agent,
                "raw_answer": raw_answer,
                **processed_answer,
            }
        return {
            "query_id": answer.query.qid,
            "agent": answer.agent,
            "raw_answer": raw_answer,
            "answer": processed_answer,
        }

    def _get_skip_tuples(self) -> set[Sequence[str]]:
        skip_tuples: set[Sequence[str]] = set()
        if self.config.force and os.path.exists(self.output_file):
            logging.warning(f"Removing existing {self.output_file}!")
            os.remove(self.output_file)

        if os.path.isfile(self.output_file):
            line: dict[str, str]
            with open(self.output_file, "r") as f:
                reader = csv.DictReader(f, fieldnames=self.output_columns)
                for line in reader:
                    skip_tuples.add(tuple(line[col] for col in self.tuple_columns))
        if len(skip_tuples) > 0:
            logging.warning(
                f"Skipping {len(skip_tuples)} games already evaluated! "
                "If you want to re-evaluate them, please use the --force flag"
            )
        return skip_tuples

    @abstractmethod
    def _build_message(self, answer) -> str:
        """Builds the message to send to the LLM evaluator"""
        raise NotImplementedError

    @abstractmethod
    def _process_answer(self, answer: str) -> str | dict[str, str]:
        """Processes the LLM evaluator output into some serializable format"""
        raise NotImplementedError

    def _dump_response(self, answer_dict: dict[str, str], file: Optional[str] = None):
        self._print_response(answer_dict)
        if not self.config.write_output:
            return
        output_file = file if file else self.output_file
        if not os.path.isfile(output_file):
            logging.debug(f"Creating new file {output_file}")
            with open(output_file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=self.output_columns)
                writer.writeheader()
        with open(output_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.output_columns)
            writer.writerow(answer_dict)

    def _print_response(self, answer_dict: dict[str, str]):
        qid = answer_dict["query_id"]
        agent = answer_dict["agent"]
        raw_answer = answer_dict["raw_answer"]
        answer_keys = set(answer_dict.keys()) - {"query_id", "agent", "raw_answer"}
        if self.config.rich_print:
            try:
                import rich

                rich.print(
                    f"[not bold white][{qid}][/not bold white] "
                    f"[bold blue]({agent})[/bold blue]"
                )
                rich.print(
                    f"[bold green]Raw Answer:[/bold green] [not bold green]{raw_answer}[/not bold green]"
                )
                for k in answer_keys:
                    rich.print(
                        f"[bold green]{k}:[/bold green] [not bold green]{answer_dict['k']}[/not bold green]"
                    )
                rich.print("")
            except ImportError:
                logging.warning("Rich not installed. Using plain print")
                self.config.rich_print = False
        else:
            tqdm.write(f"{qid} ({agent})")
            tqdm.write(f"Raw Answer: {raw_answer}")
            for k in answer_keys:
                tqdm.write(f"{k}: {answer_dict[k]}")
            tqdm.write("")

    @classmethod
    def get_config_class(cls) -> Type[BaseAnswerEvaluatorConfig]:
        return get_type_hints(cls)["config"]

    def _load_reasonings(
        self,
        reasoning_path: str,
        query_id_col: str = "query_id",
        document_id_col: str = "did",
        answer_col: str = "answer",
    ) -> dict[str, dict[str, str]]:
        reasoning: dict[str, dict[str, str]] = defaultdict(lambda: dict())
        reasoning_read = 0
        for line in csv.DictReader(open(reasoning_path)):
            reasoning_read += 1
            reasoning[line[query_id_col]][line[document_id_col]] = line[answer_col]
        logging.info(f"Loaded {reasoning_read} reasonings")
        return dict(reasoning)

    def _prepare_reasonings(self, qid: str) -> str:
        return "\n".join(
            [" ".join([f"[{idx}]", r]) for (idx, r) in self.reasonings[qid].items()]
        )


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
