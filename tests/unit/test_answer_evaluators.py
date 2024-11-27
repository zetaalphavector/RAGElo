from __future__ import annotations

import json

import pytest

from ragelo import get_answer_evaluator
from ragelo.evaluators.answer_evaluators import (
    BaseAnswerEvaluator,
    ChatPairwiseEvaluator,
    CustomPairwiseEvaluator,
    CustomPromptEvaluator,
    PairwiseAnswerEvaluator,
    PairwiseDomainExpertEvaluator,
)
from ragelo.types.evaluables import AgentAnswer, PairwiseGame
from ragelo.types.query import Query
from ragelo.types.results import AnswerEvaluatorResult


class AnswerEvaluator(BaseAnswerEvaluator):
    def _build_message(self, query: Query, answer: AgentAnswer) -> str:
        return f"Query: {query.query}\nAnswer: {answer.text}"

    def _build_message_pairwise(self, query: Query, game: PairwiseGame) -> str:
        return f"Query: {query.query}\nAnswer A: {game.agent_a_answer.text}\nAnswer B: {game.agent_b_answer.text}"


class TestAnswerEvaluator:
    def test_evaluate_single_answer(
        self,
        llm_provider_answer_mock,
        llm_provider_pairwise_answer_mock,
        experiment,
        base_answer_eval_config,
        pairwise_answer_eval_config,
    ):
        pointwise_evaluator = AnswerEvaluator.from_config(
            config=base_answer_eval_config, llm_provider=llm_provider_answer_mock
        )
        pairwise_evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_pairwise_answer_mock
        )
        query = experiment["0"]
        answer = query.answers["agent1"]
        result = pointwise_evaluator.evaluate(query, answer)
        assert isinstance(result, AnswerEvaluatorResult)
        assert isinstance(result.answer, dict)
        assert result.answer.keys() == {"quality", "trustworthiness", "originality"}
        assert result.raw_answer == '{"quality": 1, "trustworthiness": 0, "originality": 0}'
        assert result.exception is None
        assert result.pairwise is False
        assert result.qid == query.qid
        assert result.agent == answer.agent
        call_args = llm_provider_answer_mock.async_call_mocker.call_args_list
        assert len(call_args) == 1
        expected_prompt = f"Query: {query.query}\nAnswer: {answer.text}"
        assert call_args[0][0][0] == expected_prompt

        result = pairwise_evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, AnswerEvaluatorResult)
        assert isinstance(result.raw_answer, str)
        assert isinstance(result.answer, str)
        raw_answer = json.loads(result.raw_answer)
        assert raw_answer.keys() == {"answer_a_reasoning", "answer_b_reasoning", "comparison_reasoning", "winner"}
        assert result.answer == "A"
        assert result.exception is None
        assert result.pairwise is True
        assert result.qid == query.qid
        assert result.agent_a == query.answers["agent1"].agent
        assert result.agent_b == query.answers["agent2"].agent

    def test_evaluate_pairwise_game(self, llm_provider_mock, experiment, base_answer_eval_config):
        base_answer_eval_config.pairwise = True
        evaluator = AnswerEvaluator.from_config(config=base_answer_eval_config, llm_provider=llm_provider_mock)
        query = experiment["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, AnswerEvaluatorResult)
        assert result.answer == {"relevance": 1}
        assert result.raw_answer == '{"relevance": 1}'
        assert result.exception is None
        assert result.pairwise is True
        assert result.qid == query.qid
        assert result.agent_a == query.answers["agent1"].agent
        assert result.agent_b == query.answers["agent2"].agent

    def test_evaluate_experiment(self, llm_provider_mock, experiment, base_answer_eval_config):
        evaluator = AnswerEvaluator.from_config(config=base_answer_eval_config, llm_provider=llm_provider_mock)
        evaluator.evaluate_experiment(experiment)
        for query in experiment:
            for answer in query.answers.values():
                assert isinstance(answer.evaluation, AnswerEvaluatorResult)
                assert isinstance(answer.evaluation.answer, dict)
                assert isinstance(answer.evaluation.raw_answer, str)
                assert answer.evaluation.exception is None
                assert answer.evaluation.pairwise is False
                assert answer.evaluation.qid == query.qid
                assert answer.evaluation.agent == answer.agent
        base_answer_eval_config.pairwise = True
        evaluator = AnswerEvaluator.from_config(config=base_answer_eval_config, llm_provider=llm_provider_mock)
        evaluator.evaluate_experiment(experiment)
        for query in experiment:
            for game in query.pairwise_games:
                assert isinstance(game.evaluation, AnswerEvaluatorResult)
                assert isinstance(game.evaluation.answer, dict)
                assert isinstance(game.evaluation.raw_answer, str)
                assert game.evaluation.exception is None
                assert game.evaluation.pairwise is True
                assert game.evaluation.qid == query.qid
                assert game.evaluation.agent_a == game.agent_a_answer.agent
                assert game.evaluation.agent_b == game.agent_b_answer.agent


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
