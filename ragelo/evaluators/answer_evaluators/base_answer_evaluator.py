"""Base model for dealing with answer evaluators"""

import csv
import logging
import os
from abc import abstractmethod
from typing import Any, Callable, Optional, Type, get_type_hints

from tqdm import tqdm

from ragelo.evaluators.base_evaluator import BaseEvaluator
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
from ragelo.types import AgentAnswer
from ragelo.types.configurations import AnswerEvaluatorConfig


class BaseAnswerEvaluator(BaseEvaluator):
    output_columns = ["qid", "did", "raw_answer", "answer"]

    def __init__(
        self,
        config: AnswerEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        if not config.output_file:
            self.output_file = "answers_evaluator.log"
        else:
            self.output_file = config.output_file

        self.llm_provider = llm_provider
        self.config = config

    @classmethod
    def from_config(cls, config: AnswerEvaluatorConfig, llm_provider: BaseLLMProvider):
        return cls(config, llm_provider)

    @abstractmethod
    def run(self, answers: dict[str, list[AgentAnswer]]) -> list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate_single_sample(self, answer) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def _process_answer(self, answer: str) -> Any:
        """Processes the LLM evaluator output into some serializable format"""
        raise NotImplementedError

    def _dump_response(self, answer_dict: dict[str, str], file: Optional[str] = None):
        output_file = file if file else self.output_file
        if not os.path.isfile(output_file):
            logging.debug(f"Creating new file {output_file}")
            with open(output_file, "w") as f:
                writer = csv.DictWriter(f, fieldnames=self.output_columns)
                writer.writeheader()
        self._print_response(answer_dict)
        with open(output_file, "a") as f:
            writer = csv.DictWriter(f, fieldnames=self.output_columns)

    def _print_response(self, answer_dict: dict[str, str]):
        qid = answer_dict["qid"]
        agent = answer_dict["agent"]
        raw_answer = answer_dict["raw_answer"]
        answer = answer_dict["answer"]
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
                rich.print(
                    f"[bold green]Answer:[/bold green] [not bold green]{answer}[/not bold green]"
                )
                rich.print("")
            except ImportError:
                logging.warning("Rich not installed. Using plain print")
                self.config.rich_print = False
        else:
            tqdm.write(f"{qid} ({agent})")
            tqdm.write(f"Raw Answer: {raw_answer}")
            tqdm.write(f"Answer: {answer}")
            tqdm.write("")

    @classmethod
    def get_config_class(cls) -> Type[AnswerEvaluatorConfig]:
        return get_type_hints(cls)["config"]


class AnswerEvaluatorFactory:
    registry: dict[str, Type[BaseAnswerEvaluator]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
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
        llm_provider: BaseLLMProvider,
        config: Optional[AnswerEvaluatorConfig] = None,
        **kwargs,
    ):
        if evaluator_name.lower() not in cls.registry:
            raise ValueError(f"Unknown evaluator {evaluator_name}")
        if config is None:
            class_ = cls.registry[evaluator_name]
            type_config = class_.get_config_class()
            config = type_config(**kwargs)
        return cls.registry[evaluator_name.lower()].from_config(config, llm_provider)
