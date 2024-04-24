import asyncio
from typing import cast

import pytest

from ragelo import get_answer_evaluator
from ragelo.evaluators.answer_evaluators import (
    CustomPromptEvaluator,
    PairwiseWithReasoningEvaluator,
)


class TestPairwiseWithReasoningEvaluator:
    @pytest.mark.asyncio
    async def test_batch_eval(
        self,
        llm_provider_pairwise_answer_mock,
        pairwise_answer_eval_config,
        answers_test,
    ):
        evaluator = PairwiseWithReasoningEvaluator.from_config(
            config=pairwise_answer_eval_config,
            llm_provider=llm_provider_pairwise_answer_mock,
        )
        queries = await evaluator.batch_evaluate(answers_test)
        flat_answers = [(q, a) for q in queries for a in q.answers]
        flat_evaluations = [a.evaluation for (_, a) in flat_answers]
        assert all([isinstance(a, dict) for a in flat_evaluations])
        assert len(flat_answers) == 4
        assert all([isinstance(a.raw_answer, str) for a in flat_evaluations])
        assert all([isinstance(a.answer, str) for a in flat_evaluations])
        assert (
            flat_evaluations[0].answer == "A"
            and "[[A]]" in flat_evaluations[0].raw_answer
        )
        assert (
            flat_evaluations[1].answer == "B"
            and "[[B]]" in flat_evaluations[1].raw_answer
        )
        assert (
            flat_evaluations[2].answer == "C"
            and "[[C]]" in flat_evaluations[2].raw_answer
        )
        assert (
            flat_evaluations[3].answer == "C"
            and "[[C]]" in flat_evaluations[2].raw_answer
        )

        llm_call_args = llm_provider_pairwise_answer_mock.call_mocker.call_args_list
        assert len(llm_call_args) == 4
        assert isinstance(llm_call_args[0][0][0], str)
        assert llm_call_args[0][0][0] != llm_call_args[1][0][0]
        # Make sure that no games with the same agent were called
        for call_arg in llm_call_args:
            agent_a_answer = (
                call_arg[0][0]
                .split("[The Start of Assistant A's Answer]")[1]
                .split("[The End of Assistant A's Answer]")[0]
            ).strip()
            agent_b_answer = (
                call_arg[0][0]
                .split("[The Start of Assistant B's Answer]")[1]
                .split("[The End of Assistant B's Answer]")[0]
            ).strip()
            assert agent_a_answer != agent_b_answer

    def test_run(
        self,
        llm_provider_pairwise_answer_mock,
        pairwise_answer_eval_config,
        answers_test,
    ):
        pass


class TestCustomPromptEvaluator:
    def test_batch_eval(
        self,
        llm_provider_answer_mock,
        custom_answer_eval_config,
        answers_test,
    ):
        evaluator = CustomPromptEvaluator.from_config(
            config=custom_answer_eval_config,
            llm_provider=llm_provider_answer_mock,
        )
        answers = evaluator.batch_evaluate(answers_test)
        flat_answers = [(q, a) for q in answers_test for a in q.answers]
        assert all([isinstance(a.answer, dict) for a in answers])
        assert len(answers) == 4
        assert answers[0].agent == answers[2].agent == "agent1"
        assert answers[1].agent == answers[3].agent == "agent2"

        parsed_answer = cast(dict[str, int], answers[0].answer)
        assert parsed_answer["quality"] == 2
        assert parsed_answer["trustworthiness"] == 1
        assert parsed_answer["originality"] == 1

        llm_call_args = llm_provider_answer_mock.call_mocker.call_args_list
        assert (
            len(
                llm_call_args[0][0][0]
                .split("DOCUMENTS RETRIEVED:")[1]
                .split("User Query")[0]
                .strip()
                .split("\n")
            )
            == 2
        )
        for (q, a), args in zip(flat_answers, llm_call_args):
            submitted_query = args[0][0].split("User Query: ")[1].split("\n")[0].strip()
            submitted_answer = (
                args[0][0].split("Agent answer: ")[1].split("\n")[-1].strip()
            )
            expected_query = q.query
            expected_answer = a.text
            assert submitted_query == expected_query
            assert submitted_answer == expected_answer

    @pytest.mark.asyncio
    async def test_batch_eval_async(
        self,
        llm_provider_answer_mock,
        custom_answer_eval_config,
        answers_test,
    ):
        custom_answer_eval_config.n_processes = 2
        evaluator = CustomPromptEvaluator.from_config(
            config=custom_answer_eval_config,
            llm_provider=llm_provider_answer_mock,
        )

        answers = await evaluator.batch_evaluate_async(answers_test)
        flat_answers = [(q, a) for q in answers_test for a in q.answers]

        assert all([isinstance(a.answer, dict) for a in answers])
        assert len(answers) == 4
        assert answers[0].agent == answers[2].agent == "agent1"
        assert answers[1].agent == answers[3].agent == "agent2"
        assert all([x.raw_answer.startswith("Async") for x in answers])

        parsed_answer = cast(dict[str, int], answers[0].answer)
        assert parsed_answer["quality"] == 1
        assert parsed_answer["trustworthiness"] == 0
        assert parsed_answer["originality"] == 0

        llm_call_args = llm_provider_answer_mock.async_call_mocker.call_args_list
        assert (
            len(
                llm_call_args[0][0][0]
                .split("DOCUMENTS RETRIEVED:")[1]
                .split("User Query")[0]
                .strip()
                .split("\n")
            )
            == 2
        )
        for (q, a), args in zip(flat_answers, llm_call_args):
            submitted_query = args[0][0].split("User Query: ")[1].split("\n")[0].strip()
            submitted_answer = (
                args[0][0].split("Agent answer: ")[1].split("\n")[-1].strip()
            )
            expected_query = q.query
            expected_answer = a.text
            assert submitted_query == expected_query
            assert submitted_answer == expected_answer


def test_get_by_name(llm_provider_pairwise_answer_mock, pairwise_answer_eval_config):
    evaluator = get_answer_evaluator(
        "pairwise_reasoning",
        llm_provider_pairwise_answer_mock,
    )
    assert isinstance(evaluator, PairwiseWithReasoningEvaluator)
