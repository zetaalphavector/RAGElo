from __future__ import annotations

import pytest

from ragelo import get_answer_evaluator
from ragelo.evaluators.answer_evaluators import (
    ChatPairwiseEvaluator,
    CustomPairwiseEvaluator,
    CustomPromptEvaluator,
    PairwiseAnswerEvaluator,
    PairwiseDomainExpertEvaluator,
)
from ragelo.types.results import AnswerEvaluatorResult


class TestPairwiseWithReasoningEvaluator:
    @pytest.mark.asyncio
    async def test_batch_eval(
        self,
        llm_provider_pairwise_answer_mock,
        pairwise_answer_eval_config,
        answers_test,
    ):
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config,
            llm_provider=llm_provider_pairwise_answer_mock,
        )
        queries = await evaluator._evaluate_experiment_async(answers_test)
        flat_answers = [(q, a) for q in queries for a in q.pairwise_games]
        evaluations = [a.evaluation for (_, a) in flat_answers]
        assert len(evaluations) == 4
        expected_answers = ["A", "B", "C", "C"]
        for e, expected_ans in zip(evaluations, expected_answers):
            assert isinstance(e, AnswerEvaluatorResult)
            assert isinstance(e.answer, str)
            assert isinstance(e.raw_answer, str)
            assert e.answer == expected_ans
            assert f"[[{expected_ans}]]" in e.raw_answer

        llm_call_args = llm_provider_pairwise_answer_mock.async_call_mocker.call_args_list
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


class TestCustomPromptEvaluator:
    @pytest.mark.asyncio
    async def test_batch_eval(
        self,
        llm_provider_answer_mock,
        custom_answer_eval_config,
        answers_test,
    ):
        evaluator = CustomPromptEvaluator.from_config(
            config=custom_answer_eval_config,
            llm_provider=llm_provider_answer_mock,
        )
        queries = await evaluator._evaluate_experiment_async(answers_test)
        flat_answers = [(q, a) for q in queries for a in q.answers.values()]
        assert len(flat_answers) == 4

        llm_call_args = llm_provider_answer_mock.async_call_mocker.call_args_list
        assert (
            len(llm_call_args[0][0][0].split("DOCUMENTS RETRIEVED:")[1].split("User Query")[0].strip().split("\n"))
            == 2
        )
        for (q, a), args in zip(flat_answers, llm_call_args):
            evaluation = a.evaluation
            assert isinstance(evaluation, AnswerEvaluatorResult)
            assert isinstance(evaluation.answer, dict)
            assert isinstance(evaluation.raw_answer, str)
            assert isinstance(evaluation.answer["quality"], int)
            assert isinstance(evaluation.answer["trustworthiness"], int)
            assert isinstance(evaluation.answer["originality"], int)

            assert evaluation.answer["quality"] == 1
            assert evaluation.answer["trustworthiness"] == 0
            assert evaluation.answer["originality"] == 0

            submitted_query = args[0][0].split("User Query: ")[1].split("\n")[0].strip()
            submitted_answer = args[0][0].split("Agent answer: ")[1].split("\n")[-1].strip()
            expected_query = q.query
            expected_answer = a.text
            assert submitted_query == expected_query
            assert submitted_answer == expected_answer


def test_get_by_name(llm_provider_mock):
    pairwise_evaluator = get_answer_evaluator(
        "pairwise",
        llm_provider_mock,
    )
    assert isinstance(pairwise_evaluator, PairwiseAnswerEvaluator)
    custom_evaluator = get_answer_evaluator(
        "custom_prompt",
        llm_provider_mock,
    )
    assert isinstance(custom_evaluator, CustomPromptEvaluator)
    custom_pairwise_evaluator = get_answer_evaluator(
        "custom_pairwise",
        llm_provider=llm_provider_mock,
        system_prompt="system prompt",
        user_prompt="user prompt",
    )
    assert isinstance(custom_pairwise_evaluator, CustomPairwiseEvaluator)
    domain_expert_evaluator = get_answer_evaluator(
        "domain_expert",
        expert_in="computer science",
        llm_provider=llm_provider_mock,
    )
    assert isinstance(domain_expert_evaluator, PairwiseDomainExpertEvaluator)

    chat_pairwise_evaluator = get_answer_evaluator(
        "chat_pairwise",
        llm_provider=llm_provider_mock,
    )
    assert isinstance(chat_pairwise_evaluator, ChatPairwiseEvaluator)
