"""Base model for dealing with answer evaluators"""
import csv
import json
import os
import re
from abc import abstractmethod
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager, Dict, List, Set, Type

from tenacity import RetryError

from ragelo.logger import logger
from ragelo.utils.openai_client import OpenAiClient, set_credentials_from_file


@dataclass
class AnswerAnnotation:
    qid: str
    agent_a: str
    agent_b: str | None = None
    pairwise: bool = False


class AnswerEvaluator:
    def __init__(
        self,
        query_path: str,
        answers_file: str,
        output_file: str,
        evaluator_name: str,
        model_name: str = "gpt-4",
        credentials_file: str | None = None,
        verbose: bool = False,
        force: bool = False,
        **kwargs,
    ):
        self.name = evaluator_name
        self.verbose = verbose
        self.force = force
        self.agents: Set[str] = set()
        self.output_file = output_file
        self.queries = self._load_queries(query_path)
        self.answers = self._load_answers(answers_file)
        self.evaluations: List[Dict[str, str]] = []

        if credentials_file:
            set_credentials_from_file(credentials_file)

        self.openai_client = OpenAiClient(model=model_name)
        self.progress_bar: ContextManager = nullcontext()
        try:
            import rich
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeRemainingColumn,
            )

            from ragelo.utils.progress_bar import RateColumn

            self.progress_bar = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                MofNCompleteColumn(),
                RateColumn(),
                transient=True,
            )

            self.rich = True
            self.print_fn = rich.print
        except ImportError:
            self.rich = False

    def evaluate_all_answers(self) -> List[Dict[str, str]]:
        self.use_bar = self.rich
        skip_tuples = self._get_skip_tuples()

        unparsed_answers = 0
        answers = self._sample_answers()

        if self.force and os.path.isfile(self.output_file):
            logger.warning(f"Removing existing {self.output_file}!")
            os.remove(self.output_file)

        if len(answers) - len(skip_tuples) > 0:
            logger.info(f"Running {len(answers) - len(skip_tuples)} evaluations...")
            with self.progress_bar as progress:
                p_progress = (
                    progress.add_task("Evaluating answers", total=len(answers))
                    if self.use_bar and progress
                    else None
                )
                for a in answers:
                    if a in skip_tuples:
                        logger.debug(f"Skipping {a}")
                        continue
                    try:
                        annotation = self._get_annotation(a)
                        relevant = self._extract_relevant(annotation)
                    except (RetryError, ValueError):
                        continue
                    self._print_response(a, annotation, relevant)
                    self._dump_response(a, annotation, relevant)
                    if progress and p_progress:
                        progress.update(p_progress, advance=1, refresh=True)
        logger.info("✅ Done!")
        logger.info(f"Unparsed answers: {unparsed_answers}")
        logger.info(f"Total evaluations: {len(answers) - unparsed_answers}")
        return self.evaluations

    @abstractmethod
    def _build_message(self, answer: AnswerAnnotation) -> str:
        raise NotImplementedError

    @abstractmethod
    def _extract_relevant(self, answer: str) -> str:
        """Extracts the relevant part of the answer. Raises ValueError if not parsable"""
        raise NotImplementedError

    @abstractmethod
    def _check_validity(self):
        raise NotImplementedError

    @abstractmethod
    def _get_annotation(self, prompt_tuple: AnswerAnnotation) -> str:
        """Gets the annotation for a single prompt"""
        prompt = self._build_message(prompt_tuple)
        try:
            answer = self.openai_client(prompt)
            answer = self._process_answer(answer)
        except RetryError as e:
            logger.warning(f"Failed to fetch answer for {prompt_tuple}")
            raise e
        except ValueError as e:
            logger.warning(f"Failed to parse answer for {prompt_tuple}")
            raise e
        return answer

    @abstractmethod
    def _process_answer(self, answer: str) -> str:
        """Processes the answer"""
        return answer

    @abstractmethod
    def _get_skip_tuples(self) -> Set[AnswerAnnotation]:
        """Gets a set of tuples that are already annotated and should be skipped.
        If not implemented by an implementation of AnswerEvaluation, all prompts will be evaluated.
        """
        return set()

    @abstractmethod
    def _sample_answers(self) -> List[AnswerAnnotation]:
        """Sample the query and answers to be used. By default, returns all queries and answers"""
        samples = []
        for qid in self.answers:
            for agent in self.answers[qid]:
                samples.append(AnswerAnnotation(qid=qid, agent_a=agent))
        return samples

    def _print_response(
        self,
        prompt_tuple: AnswerAnnotation,
        annotation: str,
        relevant: str,
    ):
        qid = prompt_tuple.qid
        strs = [
            "[bold cyan] Query       [/bold cyan]: "
            f"[not bold cyan]{self.queries[qid]}[/not bold cyan]",
        ]

        if not prompt_tuple.pairwise:
            agent = prompt_tuple.agent_a
            strs.append(
                "[bold cyan] Agent       [/bold cyan]: "
                f"[not bold cyan]{agent}[/not bold cyan]",
            )

        else:
            strs.append(
                f"[bold blue] {prompt_tuple.agent_a} [/bold cyan] 🆚"
                f"[bold red] {prompt_tuple.agent_b}[/bold red]"
            )
            try:
                relevant = self._extract_relevant(annotation)
            except ValueError:
                return
            if relevant == "A":
                annotation = annotation.replace("[[A]]", "[bold blue]A[/bold blue]")
            elif relevant == "B":
                annotation = annotation.replace("[[B]]", "[bold red]B[/bold red]")
            else:
                annotation = annotation.replace("[[C]]", "[bold purple]C[/bold purple]")
        strs.append("[bold white] Evaluator answer: [/bold white]")
        strs.append(annotation)
        if not self.rich:
            strs = [re.sub(r"\[.*?\]", "", s) for s in strs]
        if self.verbose and logger.level > 20:  # If verbose but logger not in info mode
            for s in strs:
                if self.rich:
                    self.print_fn(s)
                else:
                    print(s)
        for s in strs:
            logger.info(s)

    def _dump_response(self, prompt: AnswerAnnotation, annotation: str, relevant: str):
        d = {
            "query_id": prompt.qid,
            "query": self.queries[prompt.qid],
            "prompt": self._build_message(prompt),
            "answer": annotation,
            "relevant": relevant,
        }
        if prompt.pairwise and prompt.agent_b:
            d["agent_a"] = prompt.agent_a
            d["agent_b"] = prompt.agent_b
        else:
            d["agent"] = prompt.agent_a
        with open(self.output_file, "a") as f:
            json.dump(d, f)
            f.write("\n")
        self.evaluations.append(d)

    def _load_queries(self, queries_path: str) -> Dict[str, str]:
        queries = {}
        if not os.path.isfile(queries_path):
            logger.exception(f"Queries file {queries_path} not found")
            raise FileNotFoundError

        for line in csv.reader(open(queries_path)):
            if "query" in line:
                continue
            qid, query = line
            queries[qid] = query
        if self.verbose:
            logger.info(f"Loaded {len(queries)} queries")
        return queries

    def _load_answers(self, answers_path: str) -> Dict[str, Dict[str, str]]:
        answers: Dict[str, Dict[str, str]] = defaultdict(lambda: dict())
        for line in csv.DictReader(open(answers_path)):
            qid = line["query_id"]
            if qid not in self.queries:
                continue
            agent = line["agent"]
            self.agents.add(agent)
            answer = line["answer"]
            answers[qid][agent] = answer
        self._check_validity()
        return answers


class AnswerEvaluatorFactory:
    registry: Dict[str, Type[AnswerEvaluator]] = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type[AnswerEvaluator]):
            if name in cls.registry:
                logger.warning(f"Overwriting {name} in registry")
            cls.registry[name.lower()] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create(cls, name: str, **kwargs) -> AnswerEvaluator:
        if name.lower() not in cls.registry:
            raise ValueError(f"Name {name} not in registry")
        return cls.registry[name.lower()](evaluator_name=name, **kwargs)
