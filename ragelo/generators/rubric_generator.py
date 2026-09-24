"""Generator that produces and revises a query's rubric: the criteria a complete answer must satisfy.

prose-check: off -- the bulk of this module is LLM prompt templates, which are data.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pydantic import Field, create_model
from pydantic.json_schema import SkipJsonSchema

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider, get_llm_provider, split_llm_provider_kwargs
from ragelo.types.answer_formats import Criterion, RubricSchema
from ragelo.types.configurations import RubricGeneratorConfig
from ragelo.types.evaluables import AgentAnswer, ChatMessage, Document
from ragelo.types.formats import LLMInputPrompt
from ragelo.types.query import Query
from ragelo.utils import call_async_fn, get_pbar, string_to_template, warn_ignored_arguments, with_guidelines

if TYPE_CHECKING:
    from ragelo.types.experiment import Experiment

logger = logging.getLogger(__name__)

CRITERION_WITHOUT_EVIDENCE = create_model(
    "Criterion", __base__=Criterion, evidence=(SkipJsonSchema[list[str]], Field(default_factory=list))
)
RUBRIC_SCHEMA_WITHOUT_EVIDENCE = create_model(
    "RubricSchema",
    __base__=RubricSchema,
    criteria=(list[CRITERION_WITHOUT_EVIDENCE], Field(description=RubricSchema.model_fields["criteria"].description)),  # type: ignore[valid-type]
)


class RubricGenerator:
    """Writes `query.rubric`, the artifact every rubric evaluator grades against."""

    documents_system_prompt = string_to_template(
        """
        {% if expert_in %}You are a domain expert in {{ expert_in }}.{% endif %}{% if company %} You work for {{ company }}.{% endif %}
        Your task is to, given a user question and a set of relevant retrieved documents, create a rubric: the criteria that a complete answer to the question must satisfy.
        Think deeply and carefully about which questions a complete and high-quality answer to the user question should answer.
        Each criterion should be a short yes/no question that can be used to evaluate whether an answer satisfies it.
        You should write {{ n_criteria }} criteria.
        {% if with_evidence %}If a criterion is supported by a document, you should include the document ID in the evidence list for that criterion.
        {% endif %}You may optionally assign a weight (a positive number) to each criterion to indicate its relative importance. More important criteria should have higher weights. If no weight is provided, all criteria are weighted equally.
        """
    )

    documents_user_prompt = string_to_template("""
        [User Question]
        {{ query.query }}

        {% if conversation_context %}
        [Conversation Context]
        {% for msg in conversation_context %}
        {{ msg }}
        {% endfor %}
        {% endif %}

        [Retrieved Documents]
        {% for doc in documents %}
        [[{{doc.did}}]] {{doc.text}}
        --------------------------------
        {% endfor %}
        """)

    reference_answer_system_prompt = string_to_template(
        """
        {% if expert_in %}You are a domain expert in {{ expert_in }}.{% endif %}{% if company %} You work for {{ company }}.{% endif %}
        Your task is to, given a user question and a known-correct answer to it, decompose that answer into a rubric: the criteria that a complete answer to the question must satisfy.
        Each criterion should be a short yes/no question about one piece of information a complete answer must contain.
        You should write at most {{ n_criteria }} criteria, and fewer if the correct answer does not support that many.
        Every criterion must be supported by the correct answer: do not add criteria for information it does not contain, and do not restate the same information as two criteria.
        {% if with_evidence %}If the correct answer attributes a piece of information to a source, include that source in the evidence list for the criterion.
        {% endif %}You may optionally assign a weight (a positive number) to each criterion to indicate its relative importance. More important criteria should have higher weights. If no weight is provided, all criteria are weighted equally.
        """
    )

    reference_answer_user_prompt = string_to_template("""
        [User Question]
        {{ query.query }}

        [Correct Answer]
        {{ query.reference_answer }}
        """)

    refine_system_prompt = string_to_template(
        """
        {% if expert_in %}You are a domain expert in {{ expert_in }}.{% endif %}{% if company %} You work for {{ company }}.{% endif %}
        Your task is to, given a user question, its current rubric and answers graded against it, revise the rubric: the criteria that a complete answer to the question must satisfy.
        A good criterion tells the better answers from the worse ones. Rewrite or drop a criterion that every answer meets or every answer fails, or that marks an answer down for something the question does not require. Add a criterion for a shortcoming the answers show and the rubric misses.
        Each criterion should be a short yes/no question that can be used to evaluate whether an answer satisfies it.
        Keep the name of every criterion you do not change, so its grades stay comparable. If the rubric needs no change, return it as it is.
        You should write at most {{ n_criteria }} criteria.
        {% if with_evidence %}If a criterion is supported by a document or by the correct answer, include that source in the evidence list for the criterion.
        {% endif %}You may optionally assign a weight (a positive number) to each criterion to indicate its relative importance. More important criteria should have higher weights. If no weight is provided, all criteria are weighted equally.
        """
    )

    refine_user_prompt = string_to_template("""
        [User Question]
        {{ query.query }}
        {% if query.reference_answer %}

        [Correct Answer]
        {{ query.reference_answer }}
        {% endif %}

        [Current Rubric]
        {% for criterion in query.rubric %}
        {{ criterion.criterion_name }}{% if criterion.weight %} (weight {{ criterion.weight }}){% endif %}: {{ criterion.short_question }}{% if with_evidence and criterion.evidence %} Evidence: {{ criterion.evidence | join(", ") }}{% endif %}
        {% endfor %}
        {% if documents %}

        [Retrieved Documents]
        {% for doc in documents %}
        [[{{doc.did}}]] {{doc.text}}
        --------------------------------
        {% endfor %}
        {% endif %}

        [Graded Answers]
        {% for answer in answers %}
        [[{{ answer.agent }}]] {{ answer.rendered_text }}
        {% for evaluation in answer.evaluations.values() if evaluation.answer.rubric_fingerprint == query.rubric_fingerprint %}
        {% for grade in evaluation.answer.criteria if grade.criterion in query.rubric %}
        - {{ grade.criterion.criterion_name }}: {{ grade.fulfillment }}. {{ grade.reasoning }}
        {% endfor %}
        {% endfor %}
        --------------------------------
        {% endfor %}
        """)

    def __init__(self, config: RubricGeneratorConfig, llm_provider: BaseLLMProvider):
        self.config = config
        self.llm_provider = llm_provider

    def generate(self, query: Query, conversation_context: list[ChatMessage] | None = None) -> list[Criterion]:
        return call_async_fn(self.generate_async, query, conversation_context)

    async def generate_async(
        self, query: Query, conversation_context: list[ChatMessage] | None = None
    ) -> list[Criterion]:
        """Generates the criteria for a single query. Does not write them to the query."""
        return await self._criteria(self._build_message(query, conversation_context), query)

    def refine(self, query: Query, answers: list[AgentAnswer] | None = None) -> list[Criterion]:
        return call_async_fn(self.refine_async, query, answers)

    async def refine_async(self, query: Query, answers: list[AgentAnswer] | None = None) -> list[Criterion]:
        """Revises the query's rubric from the answers graded against it, all of the query's answers by
        default. Returns the rubric unchanged when it needs no revision. Does not write it to the query."""
        if not query.rubric:
            raise ValueError(f"Query {query.qid} has no rubric to refine.")
        answers = list(query.answers.values()) if answers is None else answers
        return await self._criteria(self._build_refine_message(query, answers), query)

    def refine_experiment(
        self, experiment: Experiment, n_threads: int | None = None, should_save: bool = True
    ) -> None:
        """Replaces `query.rubric` with its revision for every query in the experiment that has a rubric and
        answers, keeping the replaced one in `query.rubric_history`."""
        n_threads = n_threads or self.config.n_processes
        failures: list[tuple[str, Exception]] = call_async_fn(self._refine_experiment_async, experiment, n_threads)
        if should_save:
            experiment.save()
        if failures:
            qid, error = failures[0]
            raise RuntimeError(f"{len(failures)} rubrics were not refined, first: {qid}: {error!r}.")

    async def _refine_experiment_async(self, experiment: Experiment, n_threads: int) -> list[tuple[str, Exception]]:
        queries = [q for q in experiment if q.rubric and q.answers]
        pbar = get_pbar(
            len(queries),
            self.config.rich_print,
            desc="Refining rubrics",
            disable=not self.config.use_progress_bar,
        )
        semaphore = asyncio.Semaphore(n_threads)
        failures: list[tuple[str, Exception]] = []

        async def refine_one(query: Query) -> None:
            async with semaphore:
                try:
                    query.replace_rubric(await self.refine_async(query))
                except Exception as e:  # noqa: BLE001 - collected and re-raised as one RuntimeError
                    logger.warning(f"Failed to refine the rubric of query {query.qid}: {e}")
                    failures.append((query.qid, e))
                pbar.update()

        await asyncio.gather(*(refine_one(q) for q in queries))
        pbar.close()
        return failures

    async def _criteria(self, llm_input: LLMInputPrompt, query: Query) -> list[Criterion]:
        llm_input = llm_input.model_copy(
            update={"system_prompt": with_guidelines(llm_input.system_prompt, self.config.guidelines)}
        )
        schema = RubricSchema if self.config.with_evidence else RUBRIC_SCHEMA_WITHOUT_EVIDENCE
        llm_response = await self.llm_provider.call_async(llm_input, response_schema=schema)
        rubric = llm_response.parsed_answer
        if not isinstance(rubric, RubricSchema):
            raise TypeError(f"Expected a RubricSchema for query {query.qid}, got {type(rubric)}")
        return [Criterion(**criterion.model_dump()) for criterion in rubric.criteria]

    def generate_experiment(
        self,
        experiment: Experiment,
        n_threads: int | None = None,
        force: bool = False,
        should_save: bool = True,
        conversation_contexts: dict[str, list[ChatMessage]] | None = None,
    ) -> None:
        """Writes `query.rubric` for every query in the experiment that does not have one."""
        n_threads = n_threads or self.config.n_processes
        failures: list[tuple[str, Exception]] = call_async_fn(
            self._generate_experiment_async,
            experiment,
            n_threads,
            force or self.config.force,
            conversation_contexts or {},
        )
        if should_save:
            experiment.save()
        if failures:
            qid, error = failures[0]
            raise RuntimeError(
                f"{len(failures)} queries have no rubric, first: {qid}: {error!r}. "
                "Generated rubrics are saved, so re-running only retries these."
            )

    async def _generate_experiment_async(
        self,
        experiment: Experiment,
        n_threads: int,
        force: bool,
        conversation_contexts: dict[str, list[ChatMessage]],
    ) -> list[tuple[str, Exception]]:
        queries = [q for q in experiment if force or not q.rubric]
        if not queries:
            logger.info(f"All {len(list(experiment))} queries already have a rubric.")
            return []
        pbar = get_pbar(
            len(queries),
            self.config.rich_print,
            desc="Generating rubrics",
            disable=not self.config.use_progress_bar,
        )
        semaphore = asyncio.Semaphore(n_threads)
        failures: list[tuple[str, Exception]] = []

        async def generate_one(query: Query) -> None:
            async with semaphore:
                try:
                    query.rubric = await self.generate_async(query, conversation_contexts.get(query.qid))
                except Exception as e:  # noqa: BLE001 - collected and re-raised as one RuntimeError
                    logger.warning(f"Failed to generate a rubric for query {query.qid}: {e}")
                    failures.append((query.qid, e))
                pbar.update()

        await asyncio.gather(*(generate_one(q) for q in queries))
        pbar.close()
        return failures

    def _build_message(self, query: Query, conversation_context: list[ChatMessage] | None) -> LLMInputPrompt:
        context: dict[str, Any] = {
            "expert_in": self.config.expert_in,
            "company": self.config.company,
            "n_criteria": self.config.n_criteria,
            "with_evidence": self.config.with_evidence,
            "query": query,
        }
        if self.config.source == "reference_answer":
            if not query.reference_answer:
                raise ValueError(
                    f"Query {query.qid} has no reference_answer, which the "
                    f"{self.config.source} rubric source decomposes into criteria."
                )
            system_prompt = self.reference_answer_system_prompt
            user_prompt = self.reference_answer_user_prompt
        else:
            if not query.retrieved_docs:
                raise ValueError(
                    f"Query {query.qid} has no retrieved documents, which the "
                    f"{self.config.source} rubric source derives criteria from."
                )
            system_prompt = self.documents_system_prompt
            user_prompt = self.documents_user_prompt
            context["documents"] = self._documents(query)
            context["conversation_context"] = conversation_context or []
        return LLMInputPrompt(
            system_prompt=system_prompt.render(**context),
            user_message=user_prompt.render(**context),
        )

    def _build_refine_message(self, query: Query, answers: list[AgentAnswer]) -> LLMInputPrompt:
        context: dict[str, Any] = {
            "expert_in": self.config.expert_in,
            "company": self.config.company,
            "n_criteria": self.config.n_criteria,
            "with_evidence": self.config.with_evidence,
            "query": query,
            "answers": answers,
            "documents": self._documents(query) if self.config.source == "documents" else [],
        }
        return LLMInputPrompt(
            system_prompt=self.refine_system_prompt.render(**context),
            user_message=self.refine_user_prompt.render(**context),
        )

    def _documents(self, query: Query) -> list[Document]:
        """The best-scored retrieved documents, as many as `documents_limit`."""
        return sorted(
            query.retrieved_docs.values(),
            key=lambda document: max(document.retrieved_by.values(), default=0.0),
            reverse=True,
        )[: self.config.documents_limit]


def get_rubric_generator(
    llm_provider: BaseLLMProvider | str = "openai",
    config: RubricGeneratorConfig | None = None,
    **kwargs,
) -> RubricGenerator:
    if isinstance(llm_provider, str):
        provider_kwargs, kwargs = split_llm_provider_kwargs(llm_provider, kwargs)
        llm_provider = get_llm_provider(llm_provider, **provider_kwargs)
    if config is None:
        warn_ignored_arguments("The rubric generator", RubricGeneratorConfig, kwargs, stacklevel=3)
        config = RubricGeneratorConfig(**kwargs)
    return RubricGenerator(config, llm_provider)
