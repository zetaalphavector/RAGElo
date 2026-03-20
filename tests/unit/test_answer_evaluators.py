import warnings
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field, create_model

from ragelo import get_answer_evaluator
from ragelo.evaluators.answer_evaluators import (
    BaseAnswerEvaluator,
    ChatPairwiseEvaluator,
    CustomPairwiseEvaluator,
    CustomPromptEvaluator,
    PairwiseAnswerEvaluator,
    PairwiseDomainExpertEvaluator,
    RubricPairwiseEvaluator,
    RubricPointwiseEvaluator,
)
from ragelo.types.answer_formats import (
    Criterion,
    CriterionEvaluation,
    PairwiseEvaluationAnswer,
    RubricAnswerFormat,
    RubricPointwiseAnswerFormat,
    RubricSchema,
)
from ragelo.types.evaluables import AgentAnswer
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.query import Query
from ragelo.types.results import AnswerEvaluatorResult, PairwiseGameEvaluatorResult


def test_get_by_name(llm_provider_mock):
    pairwise_evaluator = get_answer_evaluator("pairwise", llm_provider_mock)
    assert isinstance(pairwise_evaluator, PairwiseAnswerEvaluator)
    custom_evaluator = get_answer_evaluator(
        "custom_prompt",
        llm_provider_mock,
        system_prompt="system prompt",
        user_prompt="Query: {{ query.query }} Answer agent a: {{ answer.text }}",
    )
    assert isinstance(custom_evaluator, CustomPromptEvaluator)
    custom_pairwise_evaluator = get_answer_evaluator(
        "custom_pairwise",
        llm_provider=llm_provider_mock,
        system_prompt="system prompt",
        user_prompt="Query: {{ query.query }} Answer agent a: {{ game.agent_a_answer.text }} Answer agent b: {{ game.agent_b_answer.text }}",  # noqa: E501
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
    def test_evaluate_single_answer(
        self, llm_provider_answer_mock, experiment, base_answer_eval_config, answer_eval_format
    ):
        pointwise_evaluator = BaseAnswerEvaluator.from_config(
            config=base_answer_eval_config,
            llm_provider=llm_provider_answer_mock,
        )
        query = experiment["0"]
        answer = query.answers["agent1"]
        result = pointwise_evaluator.evaluate(query, answer)
        assert isinstance(result, AnswerEvaluatorResult)
        assert isinstance(result.answer, answer_eval_format)
        assert result.exception is None
        assert result.qid == query.qid
        assert result.agent == answer.agent
        call_args = llm_provider_answer_mock.async_call_mocker.call_args_list
        assert len(call_args) == 1
        expected_user_prompt = f"Query: {query.query}\nAnswer: {answer.text}"
        assert call_args[0][0][0].user_message == expected_user_prompt
        assert call_args[0][0][0].system_prompt == base_answer_eval_config.system_prompt.render()

    def test_evaluate_single_game(self, llm_provider_mock, experiment, pairwise_answer_eval_config):
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])

        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert isinstance(result.answer, PairwiseEvaluationAnswer)
        assert result.answer.winner in ["A", "B", "C"]  # Mock may return tie (C)
        assert result.answer.answer_a_analysis is not None
        assert result.answer.answer_b_analysis is not None
        assert result.answer.comparison_reasoning is not None
        assert result.exception is None
        assert result.qid == query.qid
        assert result.agent_a == query.answers["agent1"].agent
        assert result.agent_b == query.answers["agent2"].agent

    def test_evaluate_experiment(
        self, llm_provider_answer_mock, experiment, base_answer_eval_config, answer_eval_format
    ):
        evaluator = BaseAnswerEvaluator.from_config(
            config=base_answer_eval_config, llm_provider=llm_provider_answer_mock
        )
        evaluator.evaluate_experiment(experiment)
        evaluator_name = str(base_answer_eval_config.evaluator_name)
        for query in experiment:
            for answer in query.answers.values():
                # Check evaluations dict
                assert len(answer.evaluations) > 0
                assert evaluator_name in answer.evaluations
                evaluation = answer.evaluations[evaluator_name]
                assert isinstance(evaluation, AnswerEvaluatorResult)
                assert isinstance(evaluation.answer, answer_eval_format)
                assert evaluation.exception is None
                assert evaluation.qid == query.qid
                assert evaluation.agent == answer.agent

    def test_evaluate_pairwise_experiment(self, llm_provider_mock, experiment, pairwise_answer_eval_config):
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        evaluator.evaluate_experiment(experiment)
        evaluator_name = str(pairwise_answer_eval_config.evaluator_name)
        for query in experiment:
            # With 2 agents, there's only 1 pairwise game (agent1 vs agent2)
            assert len(query.pairwise_games) >= 1
            for game in query.pairwise_games.values():
                # Check evaluations dict
                assert len(game.evaluations) > 0
                assert evaluator_name in game.evaluations
                evaluation = game.evaluations[evaluator_name]
                assert isinstance(evaluation, PairwiseGameEvaluatorResult)
                assert isinstance(evaluation.answer, PairwiseEvaluationAnswer)
                assert evaluation.exception is None
                assert evaluation.qid == query.qid
                assert evaluation.agent_a == game.agent_a_answer.agent
                assert evaluation.agent_b == game.agent_b_answer.agent

    def test_evaluate_all_evaluables_pointwise(
        self, llm_provider_answer_mock, experiment, base_answer_eval_config, answer_eval_format
    ):
        evaluator = BaseAnswerEvaluator.from_config(
            config=base_answer_eval_config, llm_provider=llm_provider_answer_mock
        )
        query = experiment["0"]
        evaluator.evaluate_all_evaluables(query)
        evaluator_name = str(base_answer_eval_config.evaluator_name)
        for answer in query.answers.values():
            assert evaluator_name in answer.evaluations
            evaluation = answer.evaluations[evaluator_name]
            assert isinstance(evaluation, AnswerEvaluatorResult)
            assert isinstance(evaluation.answer, answer_eval_format)


class TestPairwiseAnswerEvaluator:
    def test_evaluate_single_game(
        self,
        llm_provider_mock,
        experiment_with_conversations_and_reasonings,
        pairwise_answer_eval_config,
    ):
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config,
            llm_provider=llm_provider_mock,
        )
        query = experiment_with_conversations_and_reasonings["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert isinstance(result.answer, PairwiseEvaluationAnswer)
        assert result.answer.winner in ["A", "B", "C"]  # Mock may return tie (C)

        llm_call_args = llm_provider_mock.async_call_mocker.call_args_list
        assert len(llm_call_args) >= 1  # May be called multiple times during evaluation
        assert isinstance(llm_call_args[0][0][0], LLMInputPrompt)
        # Make sure that no games with the same agent were called
        prompt = llm_call_args[0][0][0]
        assert prompt.user_message is not None
        agent_a_answer = (
            prompt.user_message.split("[The Start of Assistant A's Answer]")[1]
            .split("[The End of Assistant A's Answer]")[0]
            .strip()
        )
        agent_b_answer = (
            prompt.user_message.split("[The Start of Assistant B's Answer]")[1]
            .split("[The End of Assistant B's Answer]")[0]
            .strip()
        )
        assert agent_a_answer != agent_b_answer
        assert agent_a_answer == query.answers["agent1"].text
        assert agent_b_answer == query.answers["agent2"].text

    def test_evaluation_no_documents(self, llm_provider_mock, empty_experiment, pairwise_answer_eval_config):
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        empty_experiment.add_query("empty_query", "0")
        empty_experiment.add_agent_answer("answer_a", "agent1", "0")
        empty_experiment.add_agent_answer("answer_b", "agent2", "0")
        query = empty_experiment["0"]
        _ = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])


class TestCustomPromptEvaluator:
    def test_evaluate_single_answer(
        self, llm_provider_answer_mock, experiment, custom_answer_eval_config, answer_eval_format
    ):
        evaluator = CustomPromptEvaluator.from_config(
            config=custom_answer_eval_config,
            llm_provider=llm_provider_answer_mock,
        )
        query: Query = experiment["0"]
        answer: AgentAnswer = query.answers["agent1"]
        result = evaluator.evaluate(query, answer)
        assert isinstance(result, AnswerEvaluatorResult)
        assert isinstance(result.answer, answer_eval_format)
        assert isinstance(result.answer.quality, int)
        assert isinstance(result.answer.trustworthiness, int)
        assert isinstance(result.answer.originality, int)
        llm_call_args = llm_provider_answer_mock.async_call_mocker.call_args_list
        documents = list(query.retrieved_docs.values())
        expected_prompt = evaluator.user_prompt.render(
            query=query,
            answer=answer,
            documents=documents,
        )
        assert llm_call_args[0][0][0].user_message == expected_prompt


class TestChatPairwiseEvaluator:
    def test_evaluate_single_game(
        self,
        llm_provider_mock,
        experiment_with_conversations_and_reasonings,
        chat_pairwise_answer_eval_config,
    ):
        evaluator = ChatPairwiseEvaluator.from_config(
            config=chat_pairwise_answer_eval_config,
            llm_provider=llm_provider_mock,
        )
        query = experiment_with_conversations_and_reasonings["0"]

        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert isinstance(result.answer, PairwiseEvaluationAnswer)
        assert result.answer.winner in ["A", "B", "C"]  # Mock may return tie (C)
        llm_call_args = llm_provider_mock.async_call_mocker.call_args_list
        assert len(llm_call_args) >= 1  # May be called multiple times during evaluation
        prompt = llm_call_args[0][0][0]
        agent_a_answer = (
            prompt.user_message.split("[The Start of Conversation with Assistant A]")[1]
            .split("[The End of Conversation with Assistant A]")[0]
            .strip()
        )
        agent_b_answer = (
            prompt.user_message.split("[The Start of Conversation with Assistant B]")[1]
            .split("[The End of Conversation with Assistant B]")[0]
            .strip()
        )
        assert agent_a_answer == "\n".join([str(msg) for msg in query.answers["agent1"].conversation])
        assert agent_b_answer == "\n".join([str(msg) for msg in query.answers["agent2"].conversation])


class TestDomainExpertEvaluator:
    def test_evaluate_single_answer(
        self,
        llm_provider_mock,
        experiment_with_conversations_and_reasonings,
        domain_expert_answer_eval_config,
    ):
        evaluator = PairwiseDomainExpertEvaluator.from_config(
            config=domain_expert_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment_with_conversations_and_reasonings["0"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert isinstance(result.answer, PairwiseEvaluationAnswer)
        prompt = llm_provider_mock.async_call_mocker.call_args_list[0][0][0]
        assert isinstance(prompt, LLMInputPrompt)
        assert prompt.system_prompt is not None
        assert "You work for" not in prompt.system_prompt

    def test_evaluate_single_answer_with_company(
        self,
        llm_provider_mock,
        experiment_with_conversations_and_reasonings,
        domain_expert_answer_eval_config,
    ):
        domain_expert_answer_eval_config.company = "Zeta Alpha"
        evaluator = PairwiseDomainExpertEvaluator.from_config(
            config=domain_expert_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment_with_conversations_and_reasonings["0"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert isinstance(result.answer, PairwiseEvaluationAnswer)
        prompt = llm_provider_mock.async_call_mocker.call_args_list[0][0][0]
        assert prompt.system_prompt is not None
        assert "You work for" in prompt.system_prompt

    def test_evaluate_single_answer_no_documents(
        self,
        llm_provider_mock,
        experiment,
        domain_expert_answer_eval_config,
    ):
        evaluator = PairwiseDomainExpertEvaluator.from_config(
            config=domain_expert_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert isinstance(result.answer, PairwiseEvaluationAnswer)
        prompt = llm_provider_mock.async_call_mocker.call_args_list[0][0][0]
        assert prompt.system_prompt is not None
        assert "You work for" not in prompt.system_prompt


class TestFilterDocuments:
    """Tests for _filter_documents operator precedence and unbound variable fixes."""

    def test_documents_in_user_prompt_only(self, llm_provider_mock, experiment, pairwise_answer_eval_config):
        """When only user_prompt contains {{ documents }}, _filter_documents should still return documents."""
        pairwise_answer_eval_config.system_prompt = "System prompt without documents tag"
        pairwise_answer_eval_config.user_prompt = (
            "Query: {{ query.query }} {% for d in documents %}{{ d.text }}{% endfor %} "
            "{{ game.agent_a_answer.text }} {{ game.agent_b_answer.text }}"
        )
        pairwise_answer_eval_config.include_raw_documents = True
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]
        docs = evaluator._filter_documents(query)
        assert len(docs) > 0

    def test_documents_in_neither_prompt(self, llm_provider_mock, experiment, pairwise_answer_eval_config):
        """When neither prompt contains {{ documents }}, _filter_documents should return []."""
        pairwise_answer_eval_config.system_prompt = "System prompt"
        pairwise_answer_eval_config.user_prompt = (
            "Query: {{ query.query }} {{ game.agent_a_answer.text }} {{ game.agent_b_answer.text }}"
        )
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]
        docs = evaluator._filter_documents(query)
        assert docs == []

    def test_no_system_prompt(self, llm_provider_mock, experiment, pairwise_answer_eval_config):
        """When system_prompt is None, _filter_documents should not raise UnboundLocalError."""
        pairwise_answer_eval_config.system_prompt = None
        pairwise_answer_eval_config.user_prompt = (
            "Query: {{ query.query }} {% for d in documents %}{{ d.text }}{% endfor %} "
            "{{ game.agent_a_answer.text }} {{ game.agent_b_answer.text }}"
        )
        pairwise_answer_eval_config.include_raw_documents = True
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]
        docs = evaluator._filter_documents(query)
        assert len(docs) > 0


class TestBidirectionalPairwise:
    """Tests for bidirectional pairwise evaluation and winner reconciliation."""

    def _make_pairwise_response(
        self,
        winner: str,
        answer_a_analysis: str = "Analysis A",
        answer_b_analysis: str = "Analysis B",
        comparison_reasoning: str = "Comparison",
        answer_a_strengths: list[str] | None = None,
        answer_a_weaknesses: list[str] | None = None,
        answer_b_strengths: list[str] | None = None,
        answer_b_weaknesses: list[str] | None = None,
        winner_reasoning: str = "",
    ) -> LLMResponseType[PairwiseEvaluationAnswer]:
        answer = PairwiseEvaluationAnswer(
            answer_a_strengths=answer_a_strengths or [],
            answer_a_weaknesses=answer_a_weaknesses or [],
            answer_b_strengths=answer_b_strengths or [],
            answer_b_weaknesses=answer_b_weaknesses or [],
            answer_a_analysis=answer_a_analysis,
            answer_b_analysis=answer_b_analysis,
            comparison_reasoning=comparison_reasoning,
            winner_reasoning=winner_reasoning,
            winner=winner,
        )
        return LLMResponseType(raw_answer=answer.model_dump_json(), parsed_answer=answer)

    def test_both_directions_agree_a(self, llm_provider_mock, experiment, pairwise_answer_eval_config):
        """When both directions say A wins (normal=A, reversed=B meaning original A), final winner is A."""
        responses = [self._make_pairwise_response("A"), self._make_pairwise_response("B")]
        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=responses)
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert result.answer is not None
        assert result.answer.winner == "A"
        assert result.a_vs_b_result is not None
        assert result.b_vs_a_result is not None

    def test_both_directions_agree_b(self, llm_provider_mock, experiment, pairwise_answer_eval_config):
        """When both directions say B wins (normal=B, reversed=A meaning original B), final winner is B."""
        responses = [self._make_pairwise_response("B"), self._make_pairwise_response("A")]
        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=responses)
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert result.answer is not None
        assert result.answer.winner == "B"

    @pytest.mark.parametrize(
        ("forward_winner", "reversed_winner", "expected_winner"),
        [
            ("A", "B", "A"),
            ("A", "C", "A"),
            ("C", "B", "A"),
            ("A", "A", "C"),
            ("C", "C", "C"),
        ],
    )
    def test_reconciles_bidirectional_winners(
        self,
        llm_provider_mock,
        experiment,
        pairwise_answer_eval_config,
        forward_winner,
        reversed_winner,
        expected_winner,
    ):
        responses = [
            self._make_pairwise_response(forward_winner),
            self._make_pairwise_response(reversed_winner),
        ]
        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=responses)
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]

        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])

        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert result.answer is not None
        assert result.answer.winner == expected_winner
        assert result.a_vs_b_result is not None
        assert result.b_vs_a_result is not None
        assert result.a_vs_b_result.answer is not None
        assert result.b_vs_a_result.answer is not None
        assert result.a_vs_b_result.answer.winner == forward_winner
        assert result.b_vs_a_result.answer.winner == reversed_winner

    def test_disagreement_results_in_tie(self, llm_provider_mock, experiment, pairwise_answer_eval_config):
        """When directions disagree (both say A in their own frame), result is a tie."""
        responses = [self._make_pairwise_response("A"), self._make_pairwise_response("A")]
        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=responses)
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert result.answer is not None
        assert result.answer.winner == "C"

    def test_sub_results_are_stored(self, llm_provider_mock, experiment, pairwise_answer_eval_config):
        """Both directional sub-results should be stored on the parent result."""
        responses = [self._make_pairwise_response("A"), self._make_pairwise_response("B")]
        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=responses)
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert result.a_vs_b_result is not None
        assert result.b_vs_a_result is not None
        assert result.a_vs_b_result.answer is not None
        assert result.b_vs_a_result.answer is not None
        assert result.a_vs_b_result.answer.winner == "A"
        assert result.b_vs_a_result.answer.winner == "B"

    def test_uses_canonicalized_reversed_answer_when_reversed_direction_wins(
        self, llm_provider_mock, experiment, pairwise_answer_eval_config
    ):
        responses = [
            self._make_pairwise_response("C", comparison_reasoning="Forward tie"),
            self._make_pairwise_response(
                "B",
                answer_a_strengths=["[[A]] cites more relevant sources than [[B]]"],
                answer_a_weaknesses=["[[A]] omits one key fact covered by [[B]]"],
                answer_b_strengths=["[[B]] answers the question more directly than [[A]]"],
                answer_b_weaknesses=["[[B]] provides less evidence than [[A]]"],
                answer_a_analysis="Assistant A is weaker than Assistant B",
                answer_b_analysis="Assistant B is stronger than Assistant A",
                comparison_reasoning="Assistant B is better than Assistant A",
                winner_reasoning="[[B]] answers the question more directly than [[A]]",
            ),
        ]
        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=responses)
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )

        result = evaluator.evaluate(
            experiment["0"],
            answer_a=experiment["0"].answers["agent1"],
            answer_b=experiment["0"].answers["agent2"],
        )

        assert result.answer is not None
        assert result.answer.winner == "A"
        assert result.answer.answer_a_strengths == ["[[A]] answers the question more directly than [[B]]"]
        assert result.answer.answer_b_strengths == ["[[B]] cites more relevant sources than [[A]]"]
        assert result.answer.answer_a_analysis == "Assistant A is stronger than Assistant B"
        assert result.answer.answer_b_analysis == "Assistant B is weaker than Assistant A"
        assert result.answer.comparison_reasoning == "Assistant A is better than Assistant B"
        assert result.answer.winner_reasoning == "[[A]] answers the question more directly than [[B]]"


class TestPairwiseAnswerCanonicalization:
    def test_pairwise_answer_swap_perspective(self):
        answer = PairwiseEvaluationAnswer(
            answer_a_strengths=["[[A]] is more precise than [[B]]"],
            answer_a_weaknesses=["[[A]] lacks one example that [[B]] includes"],
            answer_b_strengths=["[[B]] includes one useful example for [[A]] to match"],
            answer_b_weaknesses=["[[B]] is less precise than [[A]]"],
            answer_a_analysis="Assistant A is better than Assistant B",
            answer_b_analysis="Assistant B is weaker than Assistant A",
            comparison_reasoning="Agent A is more accurate than agent B",
            winner_reasoning="[[A]] is more accurate than [[B]]",
            winner="A",
        )

        swapped = answer.swap_perspective()

        assert swapped.answer_a_strengths == ["[[A]] includes one useful example for [[B]] to match"]
        assert swapped.answer_a_weaknesses == ["[[A]] is less precise than [[B]]"]
        assert swapped.answer_b_strengths == ["[[B]] is more precise than [[A]]"]
        assert swapped.answer_b_weaknesses == ["[[B]] lacks one example that [[A]] includes"]
        assert swapped.answer_a_analysis == "Assistant A is weaker than Assistant B"
        assert swapped.answer_b_analysis == "Assistant B is better than Assistant A"
        assert swapped.comparison_reasoning == "Agent B is more accurate than agent A"
        assert swapped.winner_reasoning == "[[B]] is more accurate than [[A]]"
        assert swapped.winner == "B"

    def test_rubric_answer_swap_perspective(self):
        criterion = Criterion(criterion_name="accuracy", evidence=["doc1"], short_question="Is it accurate?")
        answer = RubricAnswerFormat(
            criteria=[
                CriterionEvaluation(
                    criterion=criterion,
                    agent_a_assessment="[[A]] addresses the core fact",
                    agent_b_assessment="[[B]] misses the core fact",
                    winner_reasoning="[[A]] is more accurate than [[B]]",
                    reasoning="Assistant A is more accurate than Assistant B",
                    winner="A",
                )
            ],
            agent_a_wins=1,
            agent_b_wins=0,
            equally_good=0,
            equally_bad=0,
            winner="A",
        )

        swapped = answer.swap_perspective()

        assert swapped.criteria[0].agent_a_assessment == "[[A]] misses the core fact"
        assert swapped.criteria[0].agent_b_assessment == "[[B]] addresses the core fact"
        assert swapped.criteria[0].winner_reasoning == "[[B]] is more accurate than [[A]]"
        assert swapped.criteria[0].reasoning == "Assistant B is more accurate than Assistant A"
        assert swapped.criteria[0].winner == "B"
        assert swapped.agent_a_wins == 0
        assert swapped.agent_b_wins == 1
        assert swapped.winner == "B"

    def test_llm_response_schema_from_config(
        self, llm_provider_answer_mock, experiment, base_answer_eval_config, answer_eval_format
    ):
        """When llm_response_schema is set on config, it should be used as the response schema."""
        evaluator = BaseAnswerEvaluator.from_config(
            config=base_answer_eval_config, llm_provider=llm_provider_answer_mock
        )
        query = experiment["0"]
        answer = query.answers["agent1"]
        result = evaluator.evaluate(query, answer)
        call_args = llm_provider_answer_mock.async_call_mocker.call_args_list
        assert call_args[0].kwargs.get("response_schema") or call_args[0][0][1] == answer_eval_format
        assert isinstance(result.answer, answer_eval_format)


class TestRubricPairwiseEvaluator:
    """Tests for RubricPairwiseEvaluator."""

    def _make_criteria(self):
        return [
            Criterion(criterion_name="accuracy", evidence=["doc1"], short_question="Is the answer accurate?"),
            Criterion(criterion_name="completeness", evidence=["doc2"], short_question="Is the answer complete?"),
            Criterion(criterion_name="clarity", evidence=[], short_question="Is the answer clear?"),
        ]

    def _make_rubric_schema(self):
        return RubricSchema(criteria=self._make_criteria())

    def _make_evaluation_response(self, winners: list[str]):
        criteria = self._make_criteria()
        EvalSchema = create_model(
            "EvaluationSchema",
            **{
                c.criterion_name: create_model(
                    c.criterion_name,
                    agent_a_assessment=(str, Field(description="agent_a_assessment")),
                    agent_b_assessment=(str, Field(description="agent_b_assessment")),
                    winner_reasoning=(str, Field(description="winner_reasoning")),
                    reasoning=(str, Field(description="reasoning")),
                    winner=(str, Field(description="winner")),
                )
                for c in criteria
            },
        )  # type: ignore[call-overload]
        data = {}
        for c, w in zip(criteria, winners):
            SubModel = EvalSchema.model_fields[c.criterion_name].annotation
            data[c.criterion_name] = SubModel(
                agent_a_assessment=f"[[A]] assessment for {c.criterion_name}",
                agent_b_assessment=f"[[B]] assessment for {c.criterion_name}",
                winner_reasoning=f"[[A]] vs [[B]] on {c.criterion_name}",
                reasoning=f"{c.criterion_name} reasoning",
                winner=w,
            )
        return EvalSchema(**data)

    def test_rubric_pairwise_with_presupplied_rubrics(self, llm_provider_mock, experiment):
        """When rubrics are pre-supplied, the evaluator skips rubric generation and only calls LLM for evaluation."""
        from ragelo.types.configurations import RubricPairwiseEvaluatorConfig

        rubrics = {"0": self._make_criteria(), "1": self._make_criteria()}
        config = RubricPairwiseEvaluatorConfig(
            expert_in="AI research",
            rubrics=rubrics,
            force=True,
        )
        eval_response = self._make_evaluation_response(["A", "B", "A"])

        responses: list[LLMResponseType] = [
            # For bidirectional: normal direction
            LLMResponseType(raw_answer="eval", parsed_answer=eval_response),
            # reversed direction
            LLMResponseType(raw_answer="eval", parsed_answer=eval_response),
        ]
        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=responses)

        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)
        assert "0" in evaluator.criteria_cache
        assert "1" in evaluator.criteria_cache

        query = experiment["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert result.answer is not None
        assert isinstance(result.answer, RubricAnswerFormat)
        assert result.answer.agent_a_wins == 2
        assert result.answer.agent_b_wins == 1

    def test_rubric_pairwise_process_answer_tallies(self, llm_provider_mock, experiment):
        """Test that _process_answer correctly tallies per-criterion winners."""
        from ragelo.types.configurations import RubricPairwiseEvaluatorConfig

        rubrics = {"0": self._make_criteria()}
        config = RubricPairwiseEvaluatorConfig(expert_in="AI", rubrics=rubrics, force=True)
        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        eval_response = self._make_evaluation_response(["A", "A", "B"])
        llm_response: LLMResponseType = LLMResponseType(raw_answer="test", parsed_answer=eval_response)
        query = experiment["0"]
        processed = evaluator._process_answer(llm_response, query)
        answer = processed.parsed_answer
        assert isinstance(answer, RubricAnswerFormat)
        assert answer.agent_a_wins == 2
        assert answer.agent_b_wins == 1
        assert answer.winner == "A"
        assert answer.criteria[0].agent_a_assessment == "[[A]] assessment for accuracy"
        assert answer.criteria[0].agent_b_assessment == "[[B]] assessment for accuracy"
        assert answer.criteria[0].winner_reasoning == "[[A]] vs [[B]] on accuracy"

    def test_rubric_pairwise_d_counted_as_equally_bad(self, llm_provider_mock, experiment):
        """Winner 'D' should be normalized to 'C' and counted as equally_bad in tally."""
        from ragelo.types.configurations import RubricPairwiseEvaluatorConfig

        rubrics = {"0": self._make_criteria()}
        config = RubricPairwiseEvaluatorConfig(expert_in="AI", rubrics=rubrics, force=True)
        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        eval_response = self._make_evaluation_response(["D", "C", "A"])
        llm_response: LLMResponseType = LLMResponseType(raw_answer="test", parsed_answer=eval_response)
        query = experiment["0"]
        processed = evaluator._process_answer(llm_response, query)
        answer = processed.parsed_answer
        assert isinstance(answer, RubricAnswerFormat)
        # D is converted to C, then "C" goes into equally_good
        # After D→C normalization: winners are [C, C, A]
        assert answer.agent_a_wins == 1
        assert answer.equally_good == 2
        assert answer.winner == "A"

    def test_build_evaluation_schema(self, llm_provider_mock):
        """_build_evaluation_schema should create a valid Pydantic model with per-criterion fields."""
        from ragelo.types.configurations import RubricPairwiseEvaluatorConfig

        config = RubricPairwiseEvaluatorConfig(expert_in="AI", force=True)
        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)
        schema = self._make_rubric_schema()
        model = evaluator._build_evaluation_schema(schema)
        assert issubclass(model, BaseModel)
        assert "accuracy" in model.model_fields
        assert "completeness" in model.model_fields
        assert "clarity" in model.model_fields


class TestRubricPointwiseEvaluator:
    """Tests for RubricPointwiseEvaluator."""

    def _make_criteria(self):
        return [
            Criterion(criterion_name="accuracy", evidence=["doc1"], short_question="Is the answer accurate?"),
            Criterion(criterion_name="completeness", evidence=["doc2"], short_question="Is the answer complete?"),
        ]

    def _make_rubric_schema(self):
        return RubricSchema(criteria=self._make_criteria())

    def test_rubric_pointwise_process_answer(self, llm_provider_mock, experiment):
        """Test that _process_answer computes average score correctly."""
        from ragelo.types.configurations import RubricPointwiseEvaluatorConfig

        config = RubricPointwiseEvaluatorConfig(expert_in="AI", force=True)
        evaluator = RubricPointwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        # Manually populate criteria cache
        criteria = self._make_criteria()
        rubric = RubricSchema(criteria=criteria)
        evaluator.criteria_cache["0"] = rubric

        EvalSchema = create_model(
            "EvaluationSchema",
            **{
                c.criterion_name: create_model(
                    c.criterion_name,
                    reasoning=(str, Field(description="reasoning")),
                    fulfillment=(bool, Field(description="fulfillment")),
                )
                for c in criteria
            },
        )  # type: ignore[call-overload]
        SubAccuracy = EvalSchema.model_fields["accuracy"].annotation
        SubCompleteness = EvalSchema.model_fields["completeness"].annotation
        eval_response = EvalSchema(
            accuracy=SubAccuracy(reasoning="acc reason", fulfillment=True),
            completeness=SubCompleteness(reasoning="comp reason", fulfillment=False),
        )
        llm_response: LLMResponseType = LLMResponseType(raw_answer="test", parsed_answer=eval_response)
        query = experiment["0"]
        processed = evaluator._process_answer(llm_response, query)
        answer = processed.parsed_answer
        assert isinstance(answer, RubricPointwiseAnswerFormat)
        assert len(answer.criteria) == 2
        assert answer.average_score == 0.5

    def test_rubric_pointwise_get_by_name(self, llm_provider_mock):
        """Rubric pointwise evaluator should be accessible via factory."""
        evaluator = get_answer_evaluator("rubric_pointwise", llm_provider_mock, expert_in="AI")
        assert isinstance(evaluator, RubricPointwiseEvaluator)

    def test_rubric_pointwise_with_presupplied_rubrics(self, llm_provider_mock, experiment):
        """When rubrics are pre-supplied, the evaluator skips rubric generation and only calls LLM for evaluation."""
        from ragelo.types.configurations import RubricPointwiseEvaluatorConfig

        criteria = self._make_criteria()
        rubrics = {"0": criteria, "1": criteria}
        config = RubricPointwiseEvaluatorConfig(expert_in="AI", rubrics=rubrics, force=True)

        EvalSchema = create_model(
            "EvaluationSchema",
            **{
                c.criterion_name: create_model(
                    c.criterion_name,
                    reasoning=(str, Field(description="reasoning")),
                    fulfillment=(bool, Field(description="fulfillment")),
                )
                for c in criteria
            },
        )  # type: ignore[call-overload]
        SubAccuracy = EvalSchema.model_fields["accuracy"].annotation
        SubCompleteness = EvalSchema.model_fields["completeness"].annotation
        eval_response = EvalSchema(
            accuracy=SubAccuracy(reasoning="acc reason", fulfillment=True),
            completeness=SubCompleteness(reasoning="comp reason", fulfillment=False),
        )
        llm_provider_mock.async_call_mocker = AsyncMock(
            side_effect=[LLMResponseType(raw_answer="eval", parsed_answer=eval_response)]
        )

        evaluator = RubricPointwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)
        assert "0" in evaluator.criteria_cache
        assert "1" in evaluator.criteria_cache

        query = experiment["0"]
        answer = query.answers["agent1"]
        result = evaluator.evaluate(query, answer)
        assert isinstance(result, AnswerEvaluatorResult)
        assert isinstance(result.answer, RubricPointwiseAnswerFormat)
        assert result.answer.average_score == 0.5
        assert len(result.answer.criteria) == 2
        # Only 1 LLM call (evaluation), no rubric generation call
        assert llm_provider_mock.async_call_mocker.call_count == 1

    def test_rubric_pairwise_get_by_name(self, llm_provider_mock):
        """Rubric pairwise evaluator should be accessible via factory."""
        evaluator = get_answer_evaluator("rubric_pairwise", llm_provider_mock, expert_in="AI")
        assert isinstance(evaluator, RubricPairwiseEvaluator)
