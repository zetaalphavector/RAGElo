import random
import re
from typing import Any, Optional

from tenacity import RetryError
from tqdm.auto import tqdm

from ragelo.evaluators.answer_evaluators.base_answer_evaluator import (
    AnswerEvaluatorFactory,
    BaseAnswerEvaluator,
)
from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.logger import logger
from ragelo.types import AgentAnswer, AnswerEvaluatorResult, AnswerEvaluatorTypes, Query
from ragelo.types.configurations import PairwiseEvaluatorConfig


@AnswerEvaluatorFactory.register(AnswerEvaluatorTypes.PAIRWISE_REASONING)
class PairwiseWithReasoningEvaluator(BaseAnswerEvaluator):
    """A evaluator that evaluates RAG-based answers pairwise, with document reasoning"""

    config: PairwiseEvaluatorConfig
    output_columns: list[str] = ["qid", "agent_a", "agent_b", "raw_answer", "answer"]
    tuple_columns: list[str] = ["qid", "agent_a", "agent_b"]
    output_file: str = "pairwise_answers_evaluations.csv"
    prompt = """
Please act as an impartial judge and evaluate the quality of the responses provided \
by two AI assistants tasked to answer the question displayed below, based on a set \
of documents retrieved by a search engine.
You should choose the assistant that best answers the user question based on a set \
of reference documents that may or not be relevant.
Answers cite documents using square brackets. For each reference document, you will \
be provided with a reasoning explaining why the document is or is not relevant.
Your evaluation should consider factors such as the correctness, helpfulness, \
completeness, accuracy, depth, and level of detail of their responses.\
Details are only useful if they answer the user question. If an answer \
contains non-relevant details, it should not be preferred over one that only \
use relevant information.
Begin your evaluation by explaining why each answer correctly answers the user \
question. Then, you should compare the two responses and provide a short explanation \
on their differences. Avoid any position biases and ensure that the order in which \
the responses were presented does not influence your decision. Do not allow the \
length of the responses to influence your evaluation. Be as objective as possible.
After providing your explanation, output your final verdict by strictly following \
this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, \
and "[[C]]" for a tie.

[User Question]
{query}

[Reference Documents]
{documents}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]
""".strip()

    def __init__(
        self,
        config: PairwiseEvaluatorConfig,
        llm_provider: BaseLLMProvider,
    ):
        super().__init__(config, llm_provider)
        if not self.config.reasoning_path:
            raise ValueError("Reasoning file is required for PairwiseWithReasoning")
        self.k = self.config.k
        self.bidirectional = self.config.bidirectional
        self.pattern = re.compile(r"\[\[([^]]+)]].*$(?:(?!\[\[).)*", re.DOTALL)

    def batch_evaluate(self, queries: list[Query]) -> list[AnswerEvaluatorResult]:
        failed_evaluations = 0
        skip_tuples = self._get_skip_tuples()
        evaluations: list[AnswerEvaluatorResult] = []
        for query in tqdm(
            queries,
            desc="Evaluating games",
            disable=not self.config.verbose,
            leave=False,
            position=0,
            ncols=100,
        ):
            games_to_play = self.__prepare_tuples_for_query(query)
            for answer_a, answer_b in tqdm(
                games_to_play,
                desc=query.qid,
                disable=not self.config.verbose,
                leave=False,
                ncols=100,
                position=1,
            ):
                agent_a = answer_a.agent
                agent_b = answer_b.agent
                if (query.qid, agent_a, agent_b) in skip_tuples:
                    logger.debug(f"Skipping {query.qid} {agent_a} {agent_b}")
                    continue
                try:
                    raw_answer, parsed_answer = self.evaluate(
                        query=query, answer_a=answer_a, answer_b=answer_b
                    )

                except (RetryError, ValueError):
                    failed_evaluations += 1
                    continue
                evaluations.append(
                    AnswerEvaluatorResult(
                        qid=query.qid,
                        agent_a=agent_a,
                        agent_b=agent_b,
                        raw_answer=raw_answer,
                        answer=parsed_answer,
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

    def _build_message(
        self, query: Query, answer: tuple[AgentAnswer, AgentAnswer]
    ) -> str:
        reasonings = self._prepare_reasonings(query.qid)
        query_metadata = self._get_usable_fields_from_metadata(
            self.prompt, query.metadata, skip_fields=[self.config.query_placeholder]
        )
        answer_a_metadata = self._get_usable_fields_from_metadata(
            self.prompt,
            answer[0].metadata,
            skip_fields=[self.config.answer_placeholder],
        )
        answer_b_metadata = self._get_usable_fields_from_metadata(
            self.prompt,
            answer[1].metadata,
            skip_fields=[self.config.answer_placeholder],
        )
        formatters = {
            self.config.query_placeholder: query.query,
            self.config.documents_placeholder: reasonings,
            "answer_a": answer[0].text,
            "answer_b": answer[1].text,
            **query_metadata,
            **answer_a_metadata,
            **answer_b_metadata,
        }
        return self.prompt.format(**formatters)

    def evaluate(
        self,
        query: Query | str,
        answer_a: AgentAnswer | str,
        answer_b: AgentAnswer | str,
        query_metadata: Optional[dict[str, Any]] = None,
        answer_a_metadata: Optional[dict[str, Any]] = None,
        answer_b_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[str, str]:
        if isinstance(query, str):
            query = Query(qid="<no_qid>", query=query)
        if isinstance(answer_a, str):
            answer_a = AgentAnswer(agent="agent_a", text=answer_a)
        if isinstance(answer_b, str):
            answer_b = AgentAnswer(agent="agent_b", text=answer_b)
        query.add_metadata(query_metadata)
        answer_a.add_metadata(answer_a_metadata)
        answer_b.add_metadata(answer_b_metadata)

        prompt = self._build_message(query, (answer_a, answer_b))
        qid = query.qid
        agent_a_id = answer_a.agent
        agent_b_id = answer_b.agent

        try:
            raw_answer = self.llm_provider(prompt)
        except RetryError as e:
            logger.warning(
                f"Failed to FETCH answers for {qid} {agent_a_id}, {agent_b_id}"
            )
            raise e
        try:
            processed_answer = self._process_answer(raw_answer)
        except ValueError as e:
            logger.warning(
                f"Failed extracting answer for {qid}, {agent_a_id}, {agent_b_id}."
                "Probably not enough tokens in the answer."
                f"Full answer:\n{raw_answer}",
            )
            raise e
        return raw_answer, processed_answer

    def __generate_games_per_query(self, query: Query) -> list[tuple[str, str]]:
        """Generates up to self.k random pairs of agents for the given query"""
        query_agents = list({x.agent for x in query.answers})
        # Create all possible pairs
        pairs = [(a, b) for a in query_agents for b in query_agents if a != b]
        if self.bidirectional:
            pairs += [(b, a) for a, b in pairs]
        random.shuffle(pairs)
        return pairs[: self.k]

    def __prepare_tuples_for_query(
        self,
        query: Query,
    ) -> list[tuple[AgentAnswer, AgentAnswer]]:
        all_tuples = []
        answers = {}
        for agent_answer in query.answers:
            answers[agent_answer.agent] = agent_answer
        random_pairs = self.__generate_games_per_query(query)
        for agent_a, agent_b in random_pairs:
            all_tuples.append((answers[agent_a], answers[agent_b]))
        return all_tuples
        # random_pairs = self.__generate_random_games(queries)

        # answers_per_agent: dict[str, dict[str, AgentAnswer]] = defaultdict(dict)
        # queries_per_agent: defaultdict[str, set] = defaultdict(set)
        # for query in queries:
        #     for agent_answer in query.answers:
        #         answers_per_agent[agent_answer.agent][query.qid] = agent_answer
        #         queries_per_agent[agent_answer.agent].add(query.qid)

        # for query in tqdm(
        #     queries,
        #     desc="Creating prompts",
        #     disable=not use_progress_bar,
        #     ncols=100,
        #     leave=False,
        #     position=0,
        # ):
        #     qid = query.qid
        #     for a, b in random_pairs:
        #         if qid not in queries_per_agent[a] or qid not in queries_per_agent[b]:
        #             continue
        #         all_tuples.append(
        #             (answers_per_agent[a][qid], answers_per_agent[b][qid])
        #         )
        # return all_tuples

    def _process_answer(self, answer: str) -> str:
        """Extracts the relevant part of an answer."""
        match_ans = self.pattern.search(answer)
        if not match_ans:
            raise ValueError(f"Could not find answer in {answer}")
        answer = match_ans.group(1)
        if answer not in ["A", "B", "C"]:
            raise ValueError(f"Unknown answer: {answer}")
        return answer
