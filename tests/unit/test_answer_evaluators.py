from __future__ import annotations

import json
import warnings


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

    def test_get_by_name(self, llm_provider_mock):
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


class TestAnswerEvaluator:
    def test_evaluate_single_answer(self, llm_provider_answer_mock, experiment, base_answer_eval_config):
        pointwise_evaluator = AnswerEvaluator.from_config(
            config=base_answer_eval_config, llm_provider=llm_provider_answer_mock
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

    def test_evaluate_single_game(self, llm_provider_pairwise_answer_mock, experiment, pairwise_answer_eval_config):
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_pairwise_answer_mock
        )
        query = experiment["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])

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

    def test_evaluate_experiment(self, llm_provider_answer_mock, experiment, base_answer_eval_config):
        evaluator = AnswerEvaluator.from_config(config=base_answer_eval_config, llm_provider=llm_provider_answer_mock)
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

    def test_evaluate_pairwise_experiment(
        self, llm_provider_pairwise_answer_mock, experiment, pairwise_answer_eval_config
    ):
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_pairwise_answer_mock
        )
        evaluator.evaluate_experiment(experiment)
        for query in experiment:
            assert len(query.pairwise_games) == 2
            for game in query.pairwise_games:
                assert isinstance(game.evaluation, AnswerEvaluatorResult)
                assert isinstance(game.evaluation.answer, str)
                assert isinstance(game.evaluation.raw_answer, str)
                assert game.evaluation.exception is None
                assert game.evaluation.pairwise is True
                assert game.evaluation.qid == query.qid
                assert game.evaluation.agent_a == game.agent_a_answer.agent
                assert game.evaluation.agent_b == game.agent_b_answer.agent


class TestPairwiseAnswerEvaluator:
    def test_evaluate_single_game(
        self,
        llm_provider_pairwise_answer_mock,
        experiment_with_conversations_and_reasonings,
        pairwise_answer_eval_config,
    ):
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config,
            llm_provider=llm_provider_pairwise_answer_mock,
        )
        query = experiment_with_conversations_and_reasonings["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, AnswerEvaluatorResult)
        assert isinstance(result.answer, str)
        assert isinstance(result.raw_answer, str)
        assert result.answer == "A"

        llm_call_args = llm_provider_pairwise_answer_mock.async_call_mocker.call_args_list
        assert len(llm_call_args) == 1
        assert isinstance(llm_call_args[0][0][0], str)
        # Make sure that no games with the same agent were called
        prompt = llm_call_args[0][0][0]
        agent_a_answer = (
            prompt.split("[The Start of Assistant A's Answer]")[1].split("[The End of Assistant A's Answer]")[0]
        ).strip()
        agent_b_answer = (
            prompt.split("[The Start of Assistant B's Answer]")[1]
            .split("[The End of Assistant B's Answer]")[0]
            .strip()
        )
        assert agent_a_answer != agent_b_answer
        assert agent_a_answer == query.answers["agent1"].text
        assert agent_b_answer == query.answers["agent2"].text

    def test_evaluation_no_documents(
        self, llm_provider_pairwise_answer_mock, empty_experiment, pairwise_answer_eval_config
    ):
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_pairwise_answer_mock
        )
        empty_experiment.add_query("empty_query", "0")
        empty_experiment.add_agent_answer("answer_a", "agent1", "0")
        empty_experiment.add_agent_answer("answer_b", "agent2", "0")
        query = empty_experiment["0"]
        _ = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])


class TestCustomPromptEvaluator:
    def test_evaluate_single_answer(self, llm_provider_answer_mock, experiment, custom_answer_eval_config):
        evaluator = CustomPromptEvaluator.from_config(
            config=custom_answer_eval_config,
            llm_provider=llm_provider_answer_mock,
        )
        query = experiment["0"]
        answer = query.answers["agent1"]
        result = evaluator.evaluate(query, answer)
        assert isinstance(result, AnswerEvaluatorResult)
        assert isinstance(result.answer, dict)
        assert isinstance(result.raw_answer, str)
        assert isinstance(result.answer["quality"], int)
        assert isinstance(result.answer["trustworthiness"], int)
        assert isinstance(result.answer["originality"], int)
        llm_call_args = llm_provider_answer_mock.async_call_mocker.call_args_list
        documents_text = []
        for did, d in query.retrieved_docs.items():
            documents_text.append(f"[{did}] {d.text}")
        expected_prompt = evaluator.prompt.format(
            query=query.query,
            answer=answer.text,
            documents="\n".join(documents_text),
        )
        assert llm_call_args[0][0][0] == expected_prompt


class TestChatPairwiseEvaluator:
    def test_evaluate_single_game(
        self,
        llm_provider_pairwise_answer_mock,
        experiment_with_conversations_and_reasonings,
        pairwise_answer_eval_config,
    ):
        evaluator = ChatPairwiseEvaluator.from_config(
            config=pairwise_answer_eval_config,
            llm_provider=llm_provider_pairwise_answer_mock,
        )
        query = experiment_with_conversations_and_reasonings["0"]

        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, AnswerEvaluatorResult)
        assert isinstance(result.answer, str)
        assert isinstance(result.raw_answer, str)
        assert result.answer == "A"
        llm_call_args = llm_provider_pairwise_answer_mock.async_call_mocker.call_args_list
        assert len(llm_call_args) == 1
        prompt = llm_call_args[0][0][0]
        ans_a = (
            prompt.split("[The Start of Conversation with Assistant A]")[1]
            .split("[The End of Conversation with Assistant A]")[0]
            .strip()
        )
        ans_b = (
            prompt.split("[The Start of Conversation with Assistant B]")[1]
            .split("[The End of Conversation with Assistant B]")[0]
            .strip()
        )
        assert ans_a == "\n".join([str(msg) for msg in query.answers["agent1"].conversation])
        assert ans_b == "\n".join([str(msg) for msg in query.answers["agent2"].conversation])


class TestDomainExpertEvaluator:
    def test_evaluate_single_answer(
        self,
        llm_provider_pairwise_answer_mock,
        experiment_with_conversations_and_reasonings,
        domain_expert_answer_eval_config,
    ):
        evaluator = PairwiseDomainExpertEvaluator.from_config(
            config=domain_expert_answer_eval_config, llm_provider=llm_provider_pairwise_answer_mock
        )
        query = experiment_with_conversations_and_reasonings["0"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, AnswerEvaluatorResult)
        prompt = llm_provider_pairwise_answer_mock.async_call_mocker.call_args_list[0][0][0]
        assert isinstance(prompt, str)
        assert "You work for" not in prompt

    def test_evaluate_single_answer_with_company(
        self,
        llm_provider_pairwise_answer_mock,
        experiment_with_conversations_and_reasonings,
        domain_expert_answer_eval_config,
    ):
        domain_expert_answer_eval_config.company = "Zeta Alpha"
        evaluator = PairwiseDomainExpertEvaluator.from_config(
            config=domain_expert_answer_eval_config, llm_provider=llm_provider_pairwise_answer_mock
        )
        query = experiment_with_conversations_and_reasonings["0"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, AnswerEvaluatorResult)
        prompt = llm_provider_pairwise_answer_mock.async_call_mocker.call_args_list[0][0][0]
        assert "You work for" in prompt

    def test_evaluate_single_answer_no_documents(
        self,
        llm_provider_pairwise_answer_mock,
        experiment,
        domain_expert_answer_eval_config,
    ):
        evaluator = PairwiseDomainExpertEvaluator.from_config(
            config=domain_expert_answer_eval_config, llm_provider=llm_provider_pairwise_answer_mock
        )
        query = experiment["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, AnswerEvaluatorResult)
        prompt = llm_provider_pairwise_answer_mock.async_call_mocker.call_args_list[0][0][0]
        assert "You work for" not in prompt
