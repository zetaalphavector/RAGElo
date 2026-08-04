import warnings
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field, ValidationError, create_model

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
from ragelo.evaluators.answer_evaluators.builtin_criteria import get_evidence_snippets
from ragelo.types.answer_formats import (
    CitationExcerptEvaluation,
    CitationQualitySchema,
    ClaimEvaluation,
    Criterion,
    CriterionEvaluation,
    CriterionEvaluationPointwise,
    EvidenceRecallSchema,
    EvidenceSnippetEvaluation,
    PairwiseEvaluationAnswer,
    RubricAnswerFormat,
    RubricPointwiseAnswerFormat,
    RubricSchema,
)
from ragelo.types.configurations import RubricPairwiseEvaluatorConfig, RubricPointwiseEvaluatorConfig
from ragelo.types.evaluables import AgentAnswer, ChatMessage, PairwiseGame
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


class TestAgentAnswer:
    def test_rendered_text_prefers_text(self):
        answer = AgentAnswer(qid="0", agent="agent1", text="Final answer")
        assert answer.rendered_text == "Final answer"

    def test_rendered_text_joins_conversation(self):
        answer = AgentAnswer(
            qid="0",
            agent="agent1",
            conversation=[
                ChatMessage(sender="User", content="What is RAG?"),
                ChatMessage(sender="Assistant", content="RAG combines retrieval and generation."),
            ],
        )
        assert answer.rendered_text == "User: What is RAG?\nAssistant: RAG combines retrieval and generation."

    def test_final_response_prefers_text(self):
        answer = AgentAnswer(qid="0", agent="agent1", text="Final answer")
        assert answer.final_response == "Final answer"

    def test_final_response_extracts_last_assistant_message(self):
        answer = AgentAnswer(
            qid="0",
            agent="agent1",
            conversation=[
                ChatMessage(sender="User", content="What is RAG?"),
                ChatMessage(sender="Assistant", content="First reply"),
                ChatMessage(sender="User", content="Tell me more"),
                ChatMessage(sender="Assistant", content="Second reply"),
            ],
        )
        assert answer.final_response == "Second reply"

    def test_final_response_falls_back_to_last_message(self):
        answer = AgentAnswer(
            qid="0",
            agent="agent1",
            conversation=[
                ChatMessage(sender="User", content="What is RAG?"),
                ChatMessage(sender="agent1", content="RAG combines retrieval and generation."),
            ],
        )
        assert answer.final_response == "RAG combines retrieval and generation."

    def test_final_response_recognizes_bot_sender(self):
        answer = AgentAnswer(
            qid="0",
            agent="agent1",
            conversation=[
                ChatMessage(sender="User", content="Hello"),
                ChatMessage(sender="Bot", content="Hi there!"),
            ],
        )
        assert answer.final_response == "Hi there!"


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
        assert swapped.criteria[0].winner == "B"
        assert swapped.agent_a_wins == 0
        assert swapped.agent_b_wins == 1
        assert swapped.winner == "B"

    def test_rubric_answer_merge_with_canonicalized(self):
        """Merging two same-perspective directions should produce consistent aggregates and per-criterion verdicts."""
        accuracy = Criterion(criterion_name="accuracy", evidence=[], short_question="Is it accurate?", weight=0.2)
        clarity = Criterion(criterion_name="clarity", evidence=[], short_question="Is it clear?", weight=0.05)

        forward = RubricAnswerFormat(
            criteria=[
                CriterionEvaluation(
                    criterion=accuracy,
                    agent_a_assessment="forward A",
                    agent_b_assessment="forward B",
                    winner_reasoning="forward acc",
                    winner="A",
                    score_a=0.8,
                    score_b=0.4,
                    confidence=0.9,
                    failure_tags=["missing_evidence"],
                ),
                CriterionEvaluation(
                    criterion=clarity,
                    agent_a_assessment="forward A clarity",
                    agent_b_assessment="forward B clarity",
                    winner_reasoning="forward clarity",
                    winner="A",
                    score_a=0.6,
                    score_b=0.5,
                    confidence=0.8,
                ),
            ],
        )
        # Reversed direction (already canonicalized to A-vs-B perspective):
        # disagrees with forward on `accuracy`, agrees on `clarity`. Weighted aggregate
        # still favors A overall because clarity remains A and accuracy collapses to a tie.
        reversed_canonical = RubricAnswerFormat(
            criteria=[
                CriterionEvaluation(
                    criterion=accuracy,
                    agent_a_assessment="reverse A",
                    agent_b_assessment="reverse B",
                    winner_reasoning="reverse acc",
                    winner="B",
                    score_a=0.5,
                    score_b=0.6,
                    confidence=0.7,
                    failure_tags=["unsupported_claim"],
                ),
                CriterionEvaluation(
                    criterion=clarity,
                    agent_a_assessment="reverse A clarity",
                    agent_b_assessment="reverse B clarity",
                    winner_reasoning="reverse clarity",
                    winner="A",
                    score_a=0.7,
                    score_b=0.4,
                    confidence=1.0,
                ),
            ],
        )

        merged = forward.merge_with_canonicalized(reversed_canonical)

        # Aggregates follow from the merged criteria: clarity stays A (weight 0.05), accuracy
        # collapses to a tie (weight 0.2), so no criterion is won by B.
        assert merged.agent_a_wins == pytest.approx(0.05)
        assert merged.agent_b_wins == pytest.approx(0.0)
        assert merged.equally_good == pytest.approx(0.2)
        assert merged.margin == pytest.approx(0.05)
        assert merged.winner == "A"
        assert merged.mean_confidence == pytest.approx(0.85)
        # Per-criterion verdicts: disagreement collapses to "C", agreement is preserved.
        merged_by_name = {c.criterion.criterion_name: c for c in merged.criteria}
        assert merged_by_name["accuracy"].winner == "C"
        assert merged_by_name["accuracy"].score_a == pytest.approx(0.65)
        assert merged_by_name["accuracy"].score_b == pytest.approx(0.50)
        assert merged_by_name["accuracy"].confidence == pytest.approx(0.80)
        # Failure tags are unioned across directions, preserving order.
        assert merged_by_name["accuracy"].failure_tags == ["missing_evidence", "unsupported_claim"]
        assert merged_by_name["clarity"].winner == "A"
        # Free-text fields remain from the forward direction.
        assert merged_by_name["accuracy"].winner_reasoning == "forward acc"

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
                    winner=(str, Field(description="winner")),
                    score_a=(float, Field(default=0.5)),
                    score_b=(float, Field(default=0.5)),
                    loser_fix=(str, Field(default="")),
                    failure_tags=(list[str], Field(default_factory=lambda: [])),
                    confidence=(float, Field(default=0.9)),
                )
                for c in criteria
            },
        )  # type: ignore[call-overload]
        data = {}
        for c, w in zip(criteria, winners):
            SubModel = EvalSchema.model_fields[c.criterion_name].annotation
            score_a = 0.8 if w == "A" else 0.3 if w == "B" else 0.5
            score_b = 0.3 if w == "A" else 0.8 if w == "B" else 0.5
            loser_fix = f"Improve on {c.criterion_name}" if w in ("A", "B") else ""
            tags = ["incomplete_coverage"] if w in ("A", "B") else []
            data[c.criterion_name] = SubModel(
                agent_a_assessment=f"[[A]] assessment for {c.criterion_name}",
                agent_b_assessment=f"[[B]] assessment for {c.criterion_name}",
                winner_reasoning=f"[[A]] vs [[B]] on {c.criterion_name}",
                winner=w,
                score_a=score_a,
                score_b=score_b,
                loser_fix=loser_fix,
                failure_tags=tags,
                confidence=0.9,
            )
        return EvalSchema(**data)

    def test_rubric_pairwise_with_presupplied_rubrics(self, llm_provider_mock, experiment):
        """When rubrics are pre-supplied, the evaluator skips rubric generation and only calls LLM for evaluation."""
        rubrics = {"0": self._make_criteria(), "1": self._make_criteria()}
        config = RubricPairwiseEvaluatorConfig(
            expert_in="AI research",
            rubrics=rubrics,
            force=True,
        )
        eval_response = self._make_evaluation_response(["A", "B", "A"])
        # Reversed direction: an unbiased judge that judges the same content the same
        # way regardless of position will produce mirrored winners (A ↔ B) when the
        # agents are swapped. Without this, the bidirectional reconciliation would
        # correctly cancel out the verdicts as position-biased.
        reversed_eval_response = self._make_evaluation_response(["B", "A", "B"])

        responses: list[LLMResponseType] = [
            # For bidirectional: normal direction
            LLMResponseType(raw_answer="eval", parsed_answer=eval_response),
            # reversed direction
            LLMResponseType(raw_answer="eval", parsed_answer=reversed_eval_response),
        ]
        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=responses)

        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        query = experiment["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])
        assert query.rubric == rubrics["0"]
        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert result.answer is not None
        assert isinstance(result.answer, RubricAnswerFormat)
        assert result.answer.agent_a_wins == 2
        assert result.answer.agent_b_wins == 1

    def test_rubric_pairwise_prompt_renders_conversations(self, llm_provider_mock, experiment):
        config = RubricPairwiseEvaluatorConfig(expert_in="AI", rubrics={"0": self._make_criteria()}, force=True)
        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)
        query = experiment["0"]
        answer_a = AgentAnswer(
            qid=query.qid,
            agent="agent1",
            conversation=[
                ChatMessage(sender="User", content="What is the capital of Brazil?"),
                ChatMessage(sender="Assistant", content="Please clarify if you want the current capital."),
                ChatMessage(sender="User", content=query.query),
                ChatMessage(sender="Assistant", content="Brasilia is the capital of Brazil."),
            ],
        )
        answer_b = AgentAnswer(
            qid=query.qid,
            agent="agent2",
            conversation=[
                ChatMessage(sender="User", content="What is the capital of Brazil?"),
                ChatMessage(sender="Assistant", content="Please clarify if you want the current capital."),
                ChatMessage(sender="User", content=query.query),
                ChatMessage(sender="Assistant", content="Rio de Janeiro used to be the capital."),
            ],
        )

        prompt = evaluator._build_message_pairwise(
            query,
            PairwiseGame(qid=query.qid, agent_a_answer=answer_a, agent_b_answer=answer_b),
        )
        assert prompt.system_prompt
        assert prompt.user_message
        assert "two conversations written by two agents" in prompt.system_prompt
        assert "User: What is the capital of Brazil?" in prompt.user_message
        assert "Assistant: Brasilia is the capital of Brazil." in prompt.user_message
        assert "Assistant: Rio de Janeiro used to be the capital." in prompt.user_message

    def test_rubric_pairwise_criteria_uses_shared_conversation_context_only(self, llm_provider_mock, experiment):
        llm_provider_mock.async_call_mocker = AsyncMock(
            side_effect=[LLMResponseType(raw_answer="criteria", parsed_answer=self._make_rubric_schema())]
        )
        evaluator = RubricPairwiseEvaluator.from_config(
            config=RubricPairwiseEvaluatorConfig(expert_in="AI", force=True),
            llm_provider=llm_provider_mock,
        )
        query = experiment["0"]
        answer_a = AgentAnswer(
            qid=query.qid,
            agent="agent1",
            conversation=[
                ChatMessage(sender="User", content="Start question"),
                ChatMessage(sender="Assistant", content="Shared follow-up context"),
                ChatMessage(sender="User", content=query.query),
                ChatMessage(sender="Assistant", content="Agent A final answer"),
            ],
        )
        answer_b = AgentAnswer(
            qid=query.qid,
            agent="agent2",
            conversation=[
                ChatMessage(sender="User", content="Start question"),
                ChatMessage(sender="Assistant", content="Shared follow-up context"),
                ChatMessage(sender="User", content=query.query),
                ChatMessage(sender="Assistant", content="Agent B final answer"),
            ],
        )

        query.add_agent_answer(answer_a, force=True)
        query.add_agent_answer(answer_b, force=True)
        evaluator.prepare_query(query)

        criteria_call = llm_provider_mock.async_call_mocker.call_args_list[0][0][0]
        assert "[Conversation Context]" in criteria_call.user_message
        assert "User: Start question" in criteria_call.user_message
        assert "Assistant: Shared follow-up context" in criteria_call.user_message
        assert f"User: {query.query}" in criteria_call.user_message
        assert "Agent A final answer" not in criteria_call.user_message
        assert "Agent B final answer" not in criteria_call.user_message

    def test_rubric_pairwise_process_answer_tallies(self, llm_provider_mock, experiment):
        """Test that _process_answer correctly tallies per-criterion winners."""

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
        """Winner 'D' should be preserved and counted as equally_bad in tally."""

        rubrics = {"0": self._make_criteria()}
        config = RubricPairwiseEvaluatorConfig(expert_in="AI", rubrics=rubrics, force=True)
        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        eval_response = self._make_evaluation_response(["D", "C", "A"])
        llm_response: LLMResponseType = LLMResponseType(raw_answer="test", parsed_answer=eval_response)
        query = experiment["0"]
        processed = evaluator._process_answer(llm_response, query)
        answer = processed.parsed_answer
        assert isinstance(answer, RubricAnswerFormat)
        assert answer.agent_a_wins == 1
        assert answer.equally_good == 1
        assert answer.equally_bad == 1
        assert answer.winner == "A"
        assert answer.criteria[0].winner == "D"

    def test_build_evaluation_schema(self, llm_provider_mock):
        """_build_evaluation_schema should create a valid Pydantic model with per-criterion fields."""

        config = RubricPairwiseEvaluatorConfig(expert_in="AI", force=True)
        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)
        model = evaluator._build_evaluation_schema(self._make_criteria())
        assert issubclass(model, BaseModel)
        assert "accuracy" in model.model_fields
        assert "completeness" in model.model_fields
        assert "clarity" in model.model_fields

    def test_rubric_pairwise_weighted_scoring(self, llm_provider_mock, experiment):
        """A single high-weight A-win should outweigh multiple low-weight B-wins."""

        criteria = [
            Criterion(criterion_name="accuracy", evidence=["doc1"], short_question="Accurate?", weight=5.0),
            Criterion(criterion_name="completeness", evidence=["doc2"], short_question="Complete?", weight=1.0),
            Criterion(criterion_name="clarity", evidence=[], short_question="Clear?", weight=1.0),
        ]
        rubrics = {"0": criteria}
        config = RubricPairwiseEvaluatorConfig(expert_in="AI", rubrics=rubrics, force=True)
        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        # A wins accuracy (weight=5), B wins completeness and clarity (weight=1 each)
        eval_response = self._make_evaluation_response(["A", "B", "B"])
        llm_response: LLMResponseType = LLMResponseType(raw_answer="test", parsed_answer=eval_response)
        query = experiment["0"]
        query.rubric = criteria
        processed = evaluator._process_answer(llm_response, query)
        answer = processed.parsed_answer
        assert isinstance(answer, RubricAnswerFormat)
        assert answer.agent_a_wins == 5.0
        assert answer.agent_b_wins == 2.0
        assert answer.winner == "A"

    def test_rubric_pairwise_richer_fields_populated(self, llm_provider_mock, experiment):
        """New diagnostic fields should be populated from LLM response."""

        rubrics = {"0": self._make_criteria()}
        config = RubricPairwiseEvaluatorConfig(expert_in="AI", rubrics=rubrics, force=True)
        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        eval_response = self._make_evaluation_response(["A", "B", "C"])
        llm_response: LLMResponseType = LLMResponseType(raw_answer="test", parsed_answer=eval_response)
        query = experiment["0"]
        processed = evaluator._process_answer(llm_response, query)
        answer = processed.parsed_answer
        assert isinstance(answer, RubricAnswerFormat)

        crit_a = answer.criteria[0]
        assert crit_a.winner == "A"
        assert crit_a.score_a == 0.8
        assert crit_a.score_b == 0.3
        assert crit_a.loser_fix == "Improve on accuracy"
        assert crit_a.failure_tags == ["incomplete_coverage"]
        assert crit_a.confidence == 0.9

        crit_c = answer.criteria[2]
        assert crit_c.winner == "C"
        assert crit_c.loser_fix == ""
        assert crit_c.failure_tags == []

    def test_rubric_pairwise_aggregates(self, llm_provider_mock, experiment):
        """Margin and mean_confidence should be correctly computed."""

        rubrics = {"0": self._make_criteria()}
        config = RubricPairwiseEvaluatorConfig(expert_in="AI", rubrics=rubrics, force=True)
        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        eval_response = self._make_evaluation_response(["A", "A", "B"])
        llm_response: LLMResponseType = LLMResponseType(raw_answer="test", parsed_answer=eval_response)
        query = experiment["0"]
        processed = evaluator._process_answer(llm_response, query)
        answer = processed.parsed_answer
        assert isinstance(answer, RubricAnswerFormat)
        assert answer.margin == 1.0  # 2 - 1
        assert answer.mean_confidence == 0.9

    def test_rubric_pairwise_swap_perspective_new_fields(self, llm_provider_mock, experiment):
        """swap_perspective should swap scores and labels in new fields."""
        from ragelo.types.configurations import RubricPairwiseEvaluatorConfig

        rubrics = {"0": self._make_criteria()}
        config = RubricPairwiseEvaluatorConfig(expert_in="AI", rubrics=rubrics, force=True)
        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        eval_response = self._make_evaluation_response(["A", "B", "C"])
        llm_response: LLMResponseType = LLMResponseType(raw_answer="test", parsed_answer=eval_response)
        query = experiment["0"]
        processed = evaluator._process_answer(llm_response, query)
        answer = processed.parsed_answer
        assert isinstance(answer, RubricAnswerFormat)
        swapped = answer.swap_perspective()
        assert swapped.margin == -answer.margin
        assert swapped.criteria[0].winner == "B"
        assert swapped.criteria[0].score_a == answer.criteria[0].score_b
        assert swapped.criteria[0].score_b == answer.criteria[0].score_a


class TestCriterion:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("accuracy", "accuracy"),
            ("Names the capital?", "Names_the_capital"),
            ("  cheque-cancellation  ", "cheque_cancellation"),
            ("2 step approval", "c_2_step_approval"),
        ],
    )
    def test_criterion_name_is_normalized_to_an_identifier(self, raw, expected):
        """The name is used as a response-schema field name and as a diversity-qrels subtopic id."""
        assert Criterion(criterion_name=raw, short_question="q").criterion_name == expected
        assert expected.isidentifier()

    def test_criterion_name_without_alphanumerics_is_rejected(self):
        with pytest.raises(ValidationError):
            Criterion(criterion_name="???", short_question="q")


class TestDerivedAggregates:
    """Aggregates are pure functions of the per-criterion judgments, not stored by the judge."""

    def _pointwise(self, *fulfillments: tuple[float, float | None]) -> RubricPointwiseAnswerFormat:
        return RubricPointwiseAnswerFormat(
            criteria=[
                CriterionEvaluationPointwise(
                    criterion=Criterion(criterion_name=f"c{i}", short_question="?", weight=weight),
                    reasoning="r",
                    fulfillment=fulfillment,
                )
                for i, (fulfillment, weight) in enumerate(fulfillments)
            ]
        )

    def test_pointwise_average_follows_the_criteria(self):
        answer = self._pointwise((1.0, 3.0), (0.0, 1.0))
        assert answer.average_score == pytest.approx(0.75)

        dropped = answer.model_copy(update={"criteria": answer.criteria[:1]})
        assert dropped.average_score == pytest.approx(1.0)

    def test_pointwise_average_of_no_criteria_is_zero(self):
        assert self._pointwise().average_score == 0.0

    def test_pairwise_aggregates_follow_the_criteria(self):
        answer = RubricAnswerFormat(
            criteria=[
                CriterionEvaluation(
                    criterion=Criterion(criterion_name="a", short_question="?", weight=3.0),
                    winner_reasoning="r",
                    winner="A",
                    confidence=0.6,
                ),
                CriterionEvaluation(
                    criterion=Criterion(criterion_name="b", short_question="?"),
                    winner_reasoning="r",
                    winner="B",
                    confidence=1.0,
                ),
                CriterionEvaluation(
                    criterion=Criterion(criterion_name="c", short_question="?"),
                    winner_reasoning="r",
                    winner="D",
                    confidence=0.8,
                ),
            ]
        )
        assert (answer.agent_a_wins, answer.agent_b_wins) == (3.0, 1.0)
        assert (answer.equally_good, answer.equally_bad) == (0.0, 1.0)
        assert answer.margin == 2.0
        assert answer.winner == "A"
        assert answer.mean_confidence == pytest.approx(0.8)

    def test_an_aggregate_stored_by_an_older_version_is_recomputed(self):
        """A payload written when the judge stored its own aggregate loads and recomputes it."""
        payload = {
            "answer_format": "rubric_pointwise",
            "criteria": [
                {
                    "criterion": {"criterion_name": "c0", "short_question": "?", "evidence": [], "weight": None},
                    "reasoning": "r",
                    "fulfillment": True,
                }
            ],
            "average_score": 0.123,
        }
        assert RubricPointwiseAnswerFormat.model_validate(payload).average_score == 1.0

    def test_the_aggregates_survive_a_serialization_round_trip(self):
        answer = self._pointwise((1.0, 3.0), (0.0, 1.0))
        assert answer.model_dump()["average_score"] == pytest.approx(0.75)
        assert RubricPointwiseAnswerFormat.model_validate(answer.model_dump()).average_score == pytest.approx(0.75)


class TestRubricArtifact:
    """The rubric an answer is graded against is generated once and persisted on the query."""

    def _make_criteria(self):
        return [
            Criterion(criterion_name="accuracy", evidence=["doc1"], short_question="Is the answer accurate?"),
            Criterion(criterion_name="completeness", evidence=["doc2"], short_question="Is the answer complete?"),
        ]

    def test_a_generated_rubric_lands_on_the_query(self, llm_provider_mock, experiment):
        criteria = self._make_criteria()
        eval_schema = create_model(
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
        judgement = LLMResponseType(
            raw_answer="eval",
            parsed_answer=eval_schema(
                accuracy={"reasoning": "yes", "fulfillment": True},
                completeness={"reasoning": "no", "fulfillment": False},
            ),
        )
        llm_provider_mock.async_call_mocker = AsyncMock(
            side_effect=[
                LLMResponseType(raw_answer="rubric", parsed_answer=RubricSchema(criteria=criteria)),
                judgement,
                judgement,
            ]
        )
        evaluator = RubricPointwiseEvaluator.from_config(
            config=RubricPointwiseEvaluatorConfig(expert_in="AI", force=True),
            llm_provider=llm_provider_mock,
        )
        query = experiment["0"]

        evaluator.evaluate_all_evaluables(query)

        assert [c.criterion_name for c in query.rubric] == ["accuracy", "completeness"]
        judged = query.answers["agent1"].evaluations["rubric_pointwise"]
        assert isinstance(judged.answer, RubricPointwiseAnswerFormat)
        assert judged.answer.rubric_fingerprint == query.rubric_fingerprint
        # One rubric generated for the query, then one judgement per answer.
        generation_prompt = llm_provider_mock.async_call_mocker.call_args_list[0][0][0]
        for document in query.retrieved_docs.values():
            assert document.text in generation_prompt.user_message
        assert llm_provider_mock.async_call_mocker.call_count == 1 + len(query.answers)

    def test_every_rubric_is_generated_before_any_judging_starts(self, llm_provider_mock, experiment):
        criteria = self._make_criteria()
        eval_schema = create_model(
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
        phases: list[str] = []

        async def respond(llm_input, response_schema):
            if response_schema is RubricSchema:
                phases.append("generate")
                return LLMResponseType(raw_answer="rubric", parsed_answer=RubricSchema(criteria=criteria))
            phases.append("judge")
            return LLMResponseType(
                raw_answer="eval",
                parsed_answer=eval_schema(
                    accuracy={"reasoning": "yes", "fulfillment": True},
                    completeness={"reasoning": "no", "fulfillment": False},
                ),
            )

        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=respond)
        evaluator = RubricPointwiseEvaluator.from_config(
            config=RubricPointwiseEvaluatorConfig(expert_in="AI", force=True),
            llm_provider=llm_provider_mock,
        )

        evaluator.evaluate_experiment(experiment)

        n_queries = len(list(experiment))
        assert phases[:n_queries] == ["generate"] * n_queries
        assert set(phases[n_queries:]) == {"judge"}
        assert all(q.rubric for q in experiment)

    def test_judging_a_rubricless_query_directly_says_how_to_get_a_rubric(self, llm_provider_mock, experiment):
        evaluator = RubricPointwiseEvaluator.from_config(
            config=RubricPointwiseEvaluatorConfig(expert_in="AI", force=True),
            llm_provider=llm_provider_mock,
        )
        query = experiment["0"]
        with pytest.raises(RuntimeError, match="prepare_query"):
            evaluator.evaluate(query, query.answers["agent1"])
        assert llm_provider_mock.async_call_mocker.call_count == 0

    def test_an_answer_judged_against_an_edited_rubric_is_re_evaluated(self, llm_provider_mock, experiment):
        evaluator = RubricPointwiseEvaluator.from_config(
            config=RubricPointwiseEvaluatorConfig(expert_in="AI"),
            llm_provider=llm_provider_mock,
        )
        query = experiment["0"]
        query.rubric = self._make_criteria()
        answer = query.answers["agent1"]
        query.add_evaluation(
            answer,
            AnswerEvaluatorResult(
                qid=query.qid,
                agent="agent1",
                evaluator_name="rubric_pointwise",
                answer=RubricPointwiseAnswerFormat(
                    criteria=[],
                    average_score=1.0,
                    rubric_fingerprint=query.rubric_fingerprint,
                ),
            ),
        )
        assert answer not in [e for _, e in evaluator._get_tuples_to_evaluate(experiment)]

        query.rubric = [*query.rubric, Criterion(criterion_name="clarity", short_question="Clear?")]
        assert answer in [e for _, e in evaluator._get_tuples_to_evaluate(experiment)]


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

        config = RubricPointwiseEvaluatorConfig(expert_in="AI", force=True)
        evaluator = RubricPointwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        criteria = self._make_criteria()
        experiment["0"].rubric = criteria

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

        query = experiment["0"]
        answer = query.answers["agent1"]
        result = evaluator.evaluate(query, answer)
        assert query.rubric == rubrics["0"]
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

    def test_rubric_pointwise_weighted_average(self, llm_provider_mock, experiment):
        """Weighted average should give more importance to high-weight criteria."""

        criteria = [
            Criterion(criterion_name="accuracy", evidence=["doc1"], short_question="Accurate?", weight=3.0),
            Criterion(criterion_name="completeness", evidence=["doc2"], short_question="Complete?", weight=1.0),
        ]
        config = RubricPointwiseEvaluatorConfig(expert_in="AI", force=True)
        evaluator = RubricPointwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        experiment["0"].rubric = criteria

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
        # accuracy (weight=3) fulfilled, completeness (weight=1) not fulfilled
        # weighted avg = (3*1 + 1*0) / (3+1) = 0.75
        eval_response = EvalSchema(
            accuracy=SubAccuracy(reasoning="acc reason", fulfillment=True),
            completeness=SubCompleteness(reasoning="comp reason", fulfillment=False),
        )
        llm_response: LLMResponseType = LLMResponseType(raw_answer="test", parsed_answer=eval_response)
        query = experiment["0"]
        processed = evaluator._process_answer(llm_response, query)
        answer = processed.parsed_answer
        assert isinstance(answer, RubricPointwiseAnswerFormat)
        assert answer.average_score == 0.75

    def test_rubric_pointwise_graduated_scoring(self, llm_provider_mock, experiment):
        """Graduated scoring normalizes integer scores to [0, 1] using max_score."""

        criteria = self._make_criteria()
        config = RubricPointwiseEvaluatorConfig(expert_in="AI", force=True, graduated_scoring=True, max_score=5)
        evaluator = RubricPointwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        experiment["0"].rubric = criteria

        EvalSchema = create_model(
            "EvaluationSchema",
            **{
                c.criterion_name: create_model(
                    c.criterion_name,
                    reasoning=(str, Field(description="reasoning")),
                    score=(int, Field(description="score", ge=0, le=5)),
                )
                for c in criteria
            },
        )  # type: ignore[call-overload]
        SubAccuracy = EvalSchema.model_fields["accuracy"].annotation
        SubCompleteness = EvalSchema.model_fields["completeness"].annotation
        # scores: accuracy=3, completeness=5, max_score=5
        # average = ((3/5) + (5/5)) / 2 = (0.6 + 1.0) / 2 = 0.8
        eval_response = EvalSchema(
            accuracy=SubAccuracy(reasoning="acc reason", score=3),
            completeness=SubCompleteness(reasoning="comp reason", score=5),
        )
        llm_response: LLMResponseType = LLMResponseType(raw_answer="test", parsed_answer=eval_response)
        query = experiment["0"]
        processed = evaluator._process_answer(llm_response, query)
        answer = processed.parsed_answer
        assert isinstance(answer, RubricPointwiseAnswerFormat)
        assert answer.average_score == pytest.approx(0.8)
        assert answer.criteria[0].fulfillment == pytest.approx(0.6)
        assert answer.criteria[1].fulfillment == pytest.approx(1.0)

    def test_rubric_pointwise_graduated_with_weights(self, llm_provider_mock, experiment):
        """Graduated scoring with weights applies both normalization and weighting."""

        criteria = [
            Criterion(criterion_name="accuracy", evidence=["doc1"], short_question="Accurate?", weight=3.0),
            Criterion(criterion_name="completeness", evidence=["doc2"], short_question="Complete?", weight=1.0),
        ]
        config = RubricPointwiseEvaluatorConfig(expert_in="AI", force=True, graduated_scoring=True, max_score=10)
        evaluator = RubricPointwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)

        experiment["0"].rubric = criteria

        EvalSchema = create_model(
            "EvaluationSchema",
            **{
                c.criterion_name: create_model(
                    c.criterion_name,
                    reasoning=(str, Field(description="reasoning")),
                    score=(int, Field(description="score", ge=0, le=10)),
                )
                for c in criteria
            },
        )  # type: ignore[call-overload]
        SubAccuracy = EvalSchema.model_fields["accuracy"].annotation
        SubCompleteness = EvalSchema.model_fields["completeness"].annotation
        # accuracy=8/10=0.8 (weight=3), completeness=2/10=0.2 (weight=1)
        # weighted avg = (0.8*3 + 0.2*1) / (3+1) = (2.4 + 0.2) / 4 = 0.65
        eval_response = EvalSchema(
            accuracy=SubAccuracy(reasoning="acc reason", score=8),
            completeness=SubCompleteness(reasoning="comp reason", score=2),
        )
        llm_response: LLMResponseType = LLMResponseType(raw_answer="test", parsed_answer=eval_response)
        query = experiment["0"]
        processed = evaluator._process_answer(llm_response, query)
        answer = processed.parsed_answer
        assert isinstance(answer, RubricPointwiseAnswerFormat)
        assert answer.average_score == pytest.approx(0.65)


class TestBuiltinCriteriaPointwise:
    """Tests for evidence recall and citation quality in RubricPointwiseEvaluator."""

    def _make_criteria(self):
        return [
            Criterion(
                criterion_name="accuracy",
                evidence=["snippet_from_doc1"],
                short_question="Is the answer accurate?",
            ),
        ]

    def _make_eval_response(self, criteria):
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
        SubModel = EvalSchema.model_fields["accuracy"].annotation
        return EvalSchema(accuracy=SubModel(reasoning="good", fulfillment=True))

    def test_pointwise_evidence_recall(self, llm_provider_mock, experiment):
        """Evidence recall should evaluate snippet presence and contribute to average_score."""

        criteria = self._make_criteria()
        config = RubricPointwiseEvaluatorConfig(
            expert_in="AI",
            rubrics={"0": criteria},
            force=True,
            evidence_recall=True,
            evidence_snippets={"0": ["snippet A", "snippet B", "snippet C"]},
            evidence_recall_weight=2.0,
        )
        eval_response = self._make_eval_response(criteria)

        recall_response = EvidenceRecallSchema(
            evaluations=[
                EvidenceSnippetEvaluation(snippet="snippet A", present=True, reasoning="found"),
                EvidenceSnippetEvaluation(snippet="snippet B", present=False, reasoning="not found"),
                EvidenceSnippetEvaluation(snippet="snippet C", present=True, reasoning="found"),
            ]
        )

        llm_provider_mock.async_call_mocker = AsyncMock(
            side_effect=[
                LLMResponseType(raw_answer="eval", parsed_answer=eval_response),
                LLMResponseType(raw_answer="recall", parsed_answer=recall_response),
            ]
        )

        evaluator = RubricPointwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)
        query = experiment["0"]
        answer = query.answers["agent1"]
        result = evaluator.evaluate(query, answer)

        assert isinstance(result, AnswerEvaluatorResult)
        assert isinstance(result.answer, RubricPointwiseAnswerFormat)
        assert result.answer.evidence_recall is not None
        assert result.answer.evidence_recall.snippets_found == 2
        assert result.answer.evidence_recall.total_snippets == 3
        assert result.answer.evidence_recall.recall == pytest.approx(2 / 3)
        # average_score: (1.0 * 1.0 + (2/3) * 2.0) / (1.0 + 2.0) = (1.0 + 4/3) / 3.0
        expected = (1.0 + (2 / 3) * 2.0) / 3.0
        assert result.answer.average_score == pytest.approx(expected)
        # The built-in scores by being a criterion, carrying its configured weight.
        recall_criterion = [c for c in result.answer.criteria if c.criterion.criterion_name == "evidence_recall"]
        assert len(recall_criterion) == 1
        assert recall_criterion[0].criterion.weight == 2.0
        assert recall_criterion[0].fulfillment == pytest.approx(2 / 3)

    def test_pointwise_citation_quality(self, llm_provider_mock, experiment):
        """Citation quality should evaluate claims/citations and contribute to average_score."""

        criteria = self._make_criteria()
        config = RubricPointwiseEvaluatorConfig(
            expert_in="AI",
            rubrics={"0": criteria},
            force=True,
            citation_quality=True,
            citation_quality_weight=1.0,
        )
        eval_response = self._make_eval_response(criteria)

        cq_response = CitationQualitySchema(
            claims=[
                ClaimEvaluation(claim="claim 1", has_citation=True),
                ClaimEvaluation(claim="claim 2", has_citation=False),
            ],
            citations=[
                CitationExcerptEvaluation(citation="[1]", has_relevant_excerpt=True),
            ],
        )

        llm_provider_mock.async_call_mocker = AsyncMock(
            side_effect=[
                LLMResponseType(raw_answer="eval", parsed_answer=eval_response),
                LLMResponseType(raw_answer="cq", parsed_answer=cq_response),
            ]
        )

        evaluator = RubricPointwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)
        query = experiment["0"]
        answer = query.answers["agent1"]
        result = evaluator.evaluate(query, answer)

        assert isinstance(result.answer, RubricPointwiseAnswerFormat)
        assert result.answer.citation_quality is not None
        assert result.answer.citation_quality.claims_with_citations_ratio == 0.5
        assert result.answer.citation_quality.citations_with_excerpts_ratio == 1.0
        # avg citation score = (0.5 + 1.0) / 2 = 0.75
        # average_score: (1.0 * 1.0 + 0.75 * 1.0) / (1.0 + 1.0) = 0.875
        assert result.answer.average_score == pytest.approx(0.875)

    def test_pointwise_disabled_by_default(self, llm_provider_mock, experiment):
        """When evidence_recall and citation_quality are False, no built-in criteria are evaluated."""

        criteria = self._make_criteria()
        config = RubricPointwiseEvaluatorConfig(expert_in="AI", rubrics={"0": criteria}, force=True)
        eval_response = self._make_eval_response(criteria)
        llm_provider_mock.async_call_mocker = AsyncMock(
            side_effect=[LLMResponseType(raw_answer="eval", parsed_answer=eval_response)]
        )

        evaluator = RubricPointwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)
        query = experiment["0"]
        answer = query.answers["agent1"]
        result = evaluator.evaluate(query, answer)

        assert isinstance(result.answer, RubricPointwiseAnswerFormat)
        assert result.answer.evidence_recall is None
        assert result.answer.citation_quality is None
        assert result.answer.average_score == 1.0
        assert llm_provider_mock.async_call_mocker.call_count == 1

    def test_evidence_snippets_fallback_to_criterion_evidence(self, llm_provider_mock, experiment):
        """When no config snippets provided, evidence_snippets comes from Criterion.evidence fields."""

        criteria = [
            Criterion(criterion_name="c1", evidence=["ev1", "ev2"], short_question="Q1?"),
            Criterion(criterion_name="c2", evidence=["ev3"], short_question="Q2?"),
        ]
        query = experiment["0"]
        query.rubric = criteria
        snippets = get_evidence_snippets(query, None)
        assert snippets == ["ev1", "ev2", "ev3"]


class TestBuiltinCriteriaPairwise:
    """Tests for evidence recall and citation quality in RubricPairwiseEvaluator."""

    def _make_criteria(self):
        return [
            Criterion(criterion_name="accuracy", evidence=["doc1"], short_question="Is the answer accurate?"),
        ]

    def test_pairwise_evidence_recall_a_wins(self, llm_provider_mock, experiment):
        """Agent A has better evidence recall, so it should get the evidence_recall criterion win."""
        criteria = self._make_criteria()
        config = RubricPairwiseEvaluatorConfig(
            expert_in="AI",
            rubrics={"0": criteria},
            force=True,
            evidence_recall=True,
            evidence_snippets={"0": ["snippet A", "snippet B"]},
            evidence_recall_weight=1.0,
        )

        # Build evaluation schema dynamically
        EvalSchema = create_model(
            "EvaluationSchema",
            **{
                c.criterion_name: create_model(
                    c.criterion_name,
                    agent_a_assessment=(str, Field(description="agent_a_assessment")),
                    agent_b_assessment=(str, Field(description="agent_b_assessment")),
                    winner_reasoning=(str, Field(description="winner_reasoning")),
                    winner=(str, Field(description="winner")),
                )
                for c in criteria
            },
        )  # type: ignore[call-overload]
        SubModel = EvalSchema.model_fields["accuracy"].annotation
        eval_response = EvalSchema(
            accuracy=SubModel(
                agent_a_assessment="good",
                agent_b_assessment="good",
                winner_reasoning="tie",
                winner="C",
            )
        )

        recall_a = EvidenceRecallSchema(
            evaluations=[
                EvidenceSnippetEvaluation(snippet="snippet A", present=True, reasoning="found"),
                EvidenceSnippetEvaluation(snippet="snippet B", present=True, reasoning="found"),
            ]
        )
        recall_b = EvidenceRecallSchema(
            evaluations=[
                EvidenceSnippetEvaluation(snippet="snippet A", present=True, reasoning="found"),
                EvidenceSnippetEvaluation(snippet="snippet B", present=False, reasoning="not found"),
            ]
        )

        llm_provider_mock.async_call_mocker = AsyncMock(
            side_effect=[
                # Bidirectional: normal direction eval
                LLMResponseType(raw_answer="eval", parsed_answer=eval_response),
                # Bidirectional: reversed direction eval
                LLMResponseType(raw_answer="eval", parsed_answer=eval_response),
                # Evidence recall for agent A
                LLMResponseType(raw_answer="recall_a", parsed_answer=recall_a),
                # Evidence recall for agent B
                LLMResponseType(raw_answer="recall_b", parsed_answer=recall_b),
            ]
        )

        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)
        query = experiment["0"]
        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])

        assert isinstance(result, PairwiseGameEvaluatorResult)
        assert isinstance(result.answer, RubricAnswerFormat)
        assert result.answer.evidence_recall_a is not None
        assert result.answer.evidence_recall_b is not None
        assert result.answer.evidence_recall_a.recall == 1.0
        assert result.answer.evidence_recall_b.recall == 0.5
        # Regular criterion: C (tie, equally_good=1). Evidence recall: A wins (weight=1).
        assert result.answer.agent_a_wins == 1.0
        assert result.answer.winner == "A"
        # The built-in wins by being a criterion, with its side scores recorded.
        recall_criterion = [c for c in result.answer.criteria if c.criterion.criterion_name == "evidence_recall"]
        assert len(recall_criterion) == 1
        assert (recall_criterion[0].winner, recall_criterion[0].score_a, recall_criterion[0].score_b) == (
            "A",
            1.0,
            0.5,
        )

    def test_pairwise_evidence_recall_uses_terminal_conversation_response(self, llm_provider_mock, experiment):
        """Evidence recall should evaluate only the last assistant turn from a conversation answer."""

        criteria = self._make_criteria()
        config = RubricPairwiseEvaluatorConfig(
            expert_in="AI",
            rubrics={"0": criteria},
            force=True,
            evidence_recall=True,
            evidence_snippets={"0": ["snippet A"]},
            evidence_recall_weight=1.0,
        )

        EvalSchema = create_model(
            "EvaluationSchema",
            **{
                c.criterion_name: create_model(
                    c.criterion_name,
                    agent_a_assessment=(str, Field(description="agent_a_assessment")),
                    agent_b_assessment=(str, Field(description="agent_b_assessment")),
                    winner_reasoning=(str, Field(description="winner_reasoning")),
                    winner=(str, Field(description="winner")),
                )
                for c in criteria
            },
        )  # type: ignore[call-overload]
        SubModel = EvalSchema.model_fields["accuracy"].annotation
        eval_response = EvalSchema(
            accuracy=SubModel(
                agent_a_assessment="good",
                agent_b_assessment="good",
                winner_reasoning="tie",
                winner="C",
            )
        )
        recall_response = EvidenceRecallSchema(
            evaluations=[
                EvidenceSnippetEvaluation(snippet="snippet A", present=True, reasoning="found"),
            ]
        )

        llm_provider_mock.async_call_mocker = AsyncMock(
            side_effect=[
                LLMResponseType(raw_answer="eval", parsed_answer=eval_response),
                LLMResponseType(raw_answer="eval", parsed_answer=eval_response),
                LLMResponseType(raw_answer="recall_a", parsed_answer=recall_response),
                LLMResponseType(raw_answer="recall_b", parsed_answer=recall_response),
            ]
        )

        evaluator = RubricPairwiseEvaluator.from_config(config=config, llm_provider=llm_provider_mock)
        query = experiment["0"]
        query.answers["agent1"] = AgentAnswer(
            qid=query.qid,
            agent="agent1",
            conversation=[
                ChatMessage(sender="User", content="Earlier question"),
                ChatMessage(sender="Assistant", content="Earlier assistant response"),
                ChatMessage(sender="User", content=query.query),
                ChatMessage(sender="Assistant", content="Terminal answer A"),
            ],
        )
        query.answers["agent2"] = AgentAnswer(
            qid=query.qid,
            agent="agent2",
            conversation=[
                ChatMessage(sender="User", content="Earlier question"),
                ChatMessage(sender="Assistant", content="Earlier assistant response"),
                ChatMessage(sender="User", content=query.query),
                ChatMessage(sender="Assistant", content="Terminal answer B"),
            ],
        )

        result = evaluator.evaluate(query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])

        assert isinstance(result, PairwiseGameEvaluatorResult)
        recall_a_prompt = llm_provider_mock.async_call_mocker.call_args_list[2][0][0].user_message
        recall_b_prompt = llm_provider_mock.async_call_mocker.call_args_list[3][0][0].user_message
        assert "Terminal answer A" in recall_a_prompt
        assert "Earlier question" not in recall_a_prompt
        assert "Earlier assistant response" not in recall_a_prompt
        assert "Terminal answer B" in recall_b_prompt
        assert "Earlier question" not in recall_b_prompt


class TestConversationSupportInPairwiseEvaluators:
    """Tests that PairwiseAnswerEvaluator and subclasses handle conversations."""

    def test_pairwise_evaluator_renders_conversations(
        self, llm_provider_mock, experiment, pairwise_answer_eval_config
    ):
        pairwise_answer_eval_config.user_prompt = None
        pairwise_answer_eval_config.system_prompt = None
        pairwise_answer_eval_config.include_relevance_reasoning = False
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]
        answer_a = AgentAnswer(
            qid=query.qid,
            agent="agent1",
            conversation=[
                ChatMessage(sender="User", content="What is the capital of Brazil?"),
                ChatMessage(sender="Assistant", content="Brasilia is the capital."),
            ],
        )
        answer_b = AgentAnswer(
            qid=query.qid,
            agent="agent2",
            conversation=[
                ChatMessage(sender="User", content="What is the capital of Brazil?"),
                ChatMessage(sender="Assistant", content="Rio used to be the capital."),
            ],
        )

        game = PairwiseGame(qid=query.qid, agent_a_answer=answer_a, agent_b_answer=answer_b)
        prompt = evaluator._build_message_pairwise(query, game)

        assert "Conversation with Assistant A" in prompt.user_message
        assert "Conversation with Assistant B" in prompt.user_message
        assert "User: What is the capital of Brazil?" in prompt.user_message
        assert "Assistant: Brasilia is the capital." in prompt.user_message
        assert "Assistant: Rio used to be the capital." in prompt.user_message
        assert "conversations" in prompt.system_prompt

    def test_pairwise_evaluator_renders_text_answers(self, llm_provider_mock, experiment, pairwise_answer_eval_config):
        pairwise_answer_eval_config.user_prompt = None
        pairwise_answer_eval_config.system_prompt = None
        pairwise_answer_eval_config.include_relevance_reasoning = False
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]

        game = PairwiseGame(
            qid=query.qid,
            agent_a_answer=query.answers["agent1"],
            agent_b_answer=query.answers["agent2"],
        )
        prompt = evaluator._build_message_pairwise(query, game)

        assert "Answer from Assistant A" in prompt.user_message
        assert "Answer from Assistant B" in prompt.user_message
        assert query.answers["agent1"].text in prompt.user_message
        assert query.answers["agent2"].text in prompt.user_message
        assert "answers" in prompt.system_prompt

    def test_domain_expert_evaluator_renders_conversations(
        self, llm_provider_mock, experiment, domain_expert_answer_eval_config
    ):
        evaluator = PairwiseDomainExpertEvaluator.from_config(
            config=domain_expert_answer_eval_config, llm_provider=llm_provider_mock
        )
        query = experiment["0"]
        answer_a = AgentAnswer(
            qid=query.qid,
            agent="agent1",
            conversation=[
                ChatMessage(sender="User", content="What is the capital of Brazil?"),
                ChatMessage(sender="Assistant", content="Brasilia is the capital."),
            ],
        )
        answer_b = AgentAnswer(
            qid=query.qid,
            agent="agent2",
            conversation=[
                ChatMessage(sender="User", content="What is the capital of Brazil?"),
                ChatMessage(sender="Assistant", content="Rio used to be the capital."),
            ],
        )

        game = PairwiseGame(qid=query.qid, agent_a_answer=answer_a, agent_b_answer=answer_b)
        prompt = evaluator._build_message_pairwise(query, game)

        assert "Conversation with Assistant A" in prompt.user_message
        assert "Conversation with Assistant B" in prompt.user_message
        assert "User: What is the capital of Brazil?" in prompt.user_message
        assert "Assistant: Brasilia is the capital." in prompt.user_message

    def test_pairwise_evaluator_get_conversation_prefix(self):
        conversation = [
            ChatMessage(sender="User", content="Q1"),
            ChatMessage(sender="Assistant", content="A1"),
            ChatMessage(sender="User", content="Q2"),
            ChatMessage(sender="Assistant", content="A2"),
        ]
        prefix = PairwiseAnswerEvaluator._get_conversation_prefix(conversation)
        assert len(prefix) == 3
        assert prefix[-1].content == "Q2"

    def test_pairwise_evaluator_get_conversation_prefix_no_user(self):
        conversation = [
            ChatMessage(sender="System", content="Setup"),
            ChatMessage(sender="Assistant", content="Response"),
        ]
        prefix = PairwiseAnswerEvaluator._get_conversation_prefix(conversation)
        assert prefix == conversation

    def test_pairwise_evaluator_get_shared_conversation_context(self, llm_provider_mock, pairwise_answer_eval_config):
        evaluator = PairwiseAnswerEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_mock
        )
        answer_a = AgentAnswer(
            qid="0",
            agent="agent1",
            conversation=[
                ChatMessage(sender="User", content="Q1"),
                ChatMessage(sender="Assistant", content="A1"),
                ChatMessage(sender="User", content="Q2"),
                ChatMessage(sender="Assistant", content="Final A"),
            ],
        )
        answer_b = AgentAnswer(qid="0", agent="agent2", text="Text answer")
        query = Query(qid="0", query="Q2")
        query.add_agent_answer(answer_a)
        query.add_agent_answer(answer_b)
        context = evaluator._get_conversation_context(query)
        assert len(context) == 3
        assert context[-1].content == "Q2"


class TestPRFixVerification:
    """Tests that verify the fixes from the PR review."""

    def test_pairwise_winner_imported_not_duplicated(self):
        """PairwiseWinner should be imported from answer_formats, not re-defined."""
        from ragelo.evaluators.answer_evaluators import base_answer_evaluator
        from ragelo.types import answer_formats

        # Should be the same type alias (imported, not duplicated)
        assert base_answer_evaluator.PairwiseWinner is answer_formats.PairwiseWinner

    def test_openai_provider_accepts_client_parameter(self):
        """OpenAIProvider should accept an optional client parameter for testability."""
        import inspect

        from ragelo.llm_providers.openai_client import OpenAIProvider

        sig = inspect.signature(OpenAIProvider.__init__)
        assert "client" in sig.parameters
        param = sig.parameters["client"]
        assert param.default is None

    def test_swap_pairwise_labels_does_not_swap_bare_a_b(self):
        """swap_pairwise_labels should not swap bare standalone 'A' or 'B' in natural prose."""
        from ragelo.types.answer_formats import swap_pairwise_labels

        # Bare "A" at word boundary in prose should NOT be swapped
        assert swap_pairwise_labels("A list of documents") == "A list of documents"
        assert swap_pairwise_labels("Apply B-type scoring") == "Apply B-type scoring"
        assert swap_pairwise_labels("This is A good answer") == "This is A good answer"

    def test_swap_pairwise_labels_still_swaps_labeled_patterns(self):
        """swap_pairwise_labels should still swap labeled A/B references."""
        from ragelo.types.answer_formats import swap_pairwise_labels

        assert swap_pairwise_labels("[[A]] is better") == "[[B]] is better"
        assert swap_pairwise_labels("[[B]] is better") == "[[A]] is better"
        assert swap_pairwise_labels("Agent A wins") == "Agent B wins"
        assert swap_pairwise_labels("agent B wins") == "agent A wins"
        assert swap_pairwise_labels("Assistant A is more precise") == "Assistant B is more precise"
        assert swap_pairwise_labels("answer A is better") == "answer B is better"
        assert swap_pairwise_labels("Response A is complete") == "Response B is complete"

    def test_rubric_template_renders_evidence_field(self, llm_provider_mock):
        """Rubric evaluator templates should reference the Criterion.evidence field, not supporting_documents."""
        from ragelo.evaluators.answer_evaluators.rubric_pairwise_evaluator import RubricPairwiseEvaluator
        from ragelo.evaluators.answer_evaluators.rubric_pointwise_evaluator import RubricPointwiseEvaluator

        # Check that the system_prompt template contains 'criteria.evidence' (the correct field)
        pairwise_source: str = getattr(RubricPairwiseEvaluator.system_prompt, "_ragelo_source")
        assert "criteria.evidence" in pairwise_source
        assert "criteria.supporting_documents" not in pairwise_source

        pointwise_source: str = getattr(RubricPointwiseEvaluator.system_prompt, "_ragelo_source")
        assert "criteria.evidence" in pointwise_source
        assert "criteria.supporting_documents" not in pointwise_source

        criterion = Criterion(criterion_name="c", evidence=["doc_1"], short_question="Good?")
        for graduated_scoring in (False, True):
            rendered = RubricPointwiseEvaluator.system_prompt.render(
                expert_in="testing",
                company=None,
                rubric=[criterion],
                graduated_scoring=graduated_scoring,
                max_score=5,
            )
            assert "doc_1" in rendered
            assert ("score from 0 to 5" in rendered) is graduated_scoring

    def test_rubric_template_renders_evidence_values(self, llm_provider_mock):
        """Verify that the evidence field actually renders into the prompt output."""
        from ragelo.evaluators.answer_evaluators.rubric_pairwise_evaluator import RubricPairwiseEvaluator

        criterion = Criterion(
            criterion_name="test_crit",
            evidence=["doc_1", "doc_2"],
            short_question="Is the answer good?",
        )
        rendered = RubricPairwiseEvaluator.system_prompt.render(
            expert_in="testing",
            rubric=[criterion],
            company=None,
            include_evidence=False,
            is_conversation=False,
            evidence_snippets=[],
            preserve_d=True,
            rich_output=False,
        )
        assert "doc_1" in rendered
        assert "doc_2" in rendered

    def test_shared_config_fields_inherited(self):
        """RubricPairwiseEvaluatorConfig and RubricPointwiseEvaluatorConfig should share evidence/citation fields."""
        from ragelo.types.configurations.answer_evaluator_configs import RubricEvaluatorConfigBase

        assert issubclass(RubricPairwiseEvaluatorConfig, RubricEvaluatorConfigBase)
        assert issubclass(RubricPointwiseEvaluatorConfig, RubricEvaluatorConfigBase)

        shared_fields = {
            "evidence_recall",
            "citation_quality",
            "evidence_snippets",
            "evidence_recall_weight",
            "citation_quality_weight",
            "n_criteria",
            "rubrics",
        }
        mixin_fields = set(RubricEvaluatorConfigBase.model_fields.keys())
        for field in shared_fields:
            assert field in mixin_fields, f"{field} not in RubricEvaluatorConfigBase"

    def test_shared_config_defaults_consistent(self):
        """Both rubric configs should inherit the same defaults for shared fields."""
        pairwise_config = RubricPairwiseEvaluatorConfig(expert_in="test")
        pointwise_config = RubricPointwiseEvaluatorConfig(expert_in="test")

        assert pairwise_config.evidence_recall == pointwise_config.evidence_recall == False  # noqa: E712
        assert pairwise_config.citation_quality == pointwise_config.citation_quality == False  # noqa: E712
        assert pairwise_config.evidence_recall_weight == pointwise_config.evidence_recall_weight == 1.0
        assert pairwise_config.citation_quality_weight == pointwise_config.citation_quality_weight == 1.0
        assert pairwise_config.n_criteria == pointwise_config.n_criteria == 5
        assert pairwise_config.rubrics is None
        assert pointwise_config.rubrics is None
