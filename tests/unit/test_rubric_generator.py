from unittest.mock import AsyncMock

import pytest

from ragelo.generators import RubricGenerator, get_rubric_generator
from ragelo.types.answer_formats import (
    Criterion,
    CriterionEvaluationPointwise,
    RubricPointwiseAnswerFormat,
    RubricSchema,
)
from ragelo.types.configurations import RubricGeneratorConfig
from ragelo.types.evaluables import AgentAnswer, ChatMessage, Document
from ragelo.types.formats import LLMResponseType
from ragelo.types.query import Query
from ragelo.types.results import AnswerEvaluatorResult


def _criteria() -> list[Criterion]:
    return [
        Criterion(criterion_name="capital", short_question="Does it name the capital?"),
        Criterion(criterion_name="since_when", short_question="Does it say since when?", weight=2.0),
    ]


def _responds_with_criteria(llm_provider_mock) -> None:
    llm_provider_mock.async_call_mocker = AsyncMock(
        side_effect=[LLMResponseType(raw_answer="rubric", parsed_answer=RubricSchema(criteria=_criteria()))]
    )


class TestRubricGenerator:
    def test_generates_from_the_retrieved_documents(self, llm_provider_mock, experiment):
        _responds_with_criteria(llm_provider_mock)
        generator = RubricGenerator(RubricGeneratorConfig(expert_in="geography"), llm_provider_mock)

        query = experiment["0"]
        rubric = generator.generate(query)

        assert [c.criterion_name for c in rubric] == ["capital", "since_when"]
        prompt = llm_provider_mock.async_call_mocker.call_args_list[0][0][0]
        assert "geography" in prompt.system_prompt
        for did, document in query.retrieved_docs.items():
            assert f"[[{did}]]" in prompt.user_message
            assert document.text in prompt.user_message

    def test_evidence_is_requested_only_when_configured(self, llm_provider_mock, experiment):
        _responds_with_criteria(llm_provider_mock)
        query = experiment["0"]
        RubricGenerator(RubricGeneratorConfig(expert_in="geography"), llm_provider_mock).generate(query)
        prompt, schema = llm_provider_mock.async_call_mocker.call_args_list[0][0]
        assert "evidence list" in prompt.system_prompt
        assert "report" not in prompt.system_prompt
        assert "evidence" in schema.model_json_schema()["$defs"]["Criterion"]["properties"]

        _responds_with_criteria(llm_provider_mock)
        rubric = RubricGenerator(
            RubricGeneratorConfig(expert_in="geography", with_evidence=False), llm_provider_mock
        ).generate(query)
        prompt, schema = llm_provider_mock.async_call_mocker.call_args_list[0][0]
        assert "evidence" not in prompt.system_prompt
        assert issubclass(schema, RubricSchema)
        assert "evidence" not in schema.model_json_schema()["$defs"]["Criterion"]["properties"]
        assert all(type(criterion) is Criterion for criterion in rubric)

    def test_documents_source_shows_only_the_best_scored_documents(self, llm_provider_mock, experiment):
        _responds_with_criteria(llm_provider_mock)
        generator = RubricGenerator(RubricGeneratorConfig(expert_in="geography", documents_limit=1), llm_provider_mock)
        query = experiment["0"]
        query.retrieved_docs["0"].retrieved_by = {"agent1": 1.0}
        query.retrieved_docs["1"].retrieved_by = {"agent1": 2.0}

        generator.generate(query)

        prompt = llm_provider_mock.async_call_mocker.call_args_list[0][0][0]
        assert "[[1]]" in prompt.user_message
        assert "[[0]]" not in prompt.user_message

    def test_generates_from_the_reference_answer_without_showing_the_documents(self, llm_provider_mock, experiment):
        _responds_with_criteria(llm_provider_mock)
        generator = RubricGenerator(
            RubricGeneratorConfig(expert_in="geography", source="reference_answer"), llm_provider_mock
        )
        query = experiment["0"]
        query.reference_answer = "Brasilia, since 1960."

        rubric = generator.generate(query)

        assert [c.criterion_name for c in rubric] == ["capital", "since_when"]
        prompt = llm_provider_mock.async_call_mocker.call_args_list[0][0][0]
        assert "Brasilia, since 1960." in prompt.user_message
        for document in query.retrieved_docs.values():
            assert document.text not in prompt.user_message

    def test_conversation_context_reaches_the_documents_prompt(self, llm_provider_mock, experiment):
        _responds_with_criteria(llm_provider_mock)
        generator = RubricGenerator(RubricGeneratorConfig(expert_in="geography"), llm_provider_mock)

        generator.generate(experiment["0"], [ChatMessage(sender="User", content="Shared context")])

        prompt = llm_provider_mock.async_call_mocker.call_args_list[0][0][0]
        assert "[Conversation Context]" in prompt.user_message
        assert "User: Shared context" in prompt.user_message

    @pytest.mark.parametrize(
        "source,query_kwargs,missing",
        [
            ("reference_answer", {}, "reference_answer"),
            ("documents", {}, "retrieved documents"),
        ],
    )
    def test_raises_when_the_source_it_derives_from_is_missing(self, llm_provider_mock, source, query_kwargs, missing):
        generator = RubricGenerator(RubricGeneratorConfig(expert_in="geography", source=source), llm_provider_mock)
        with pytest.raises(ValueError, match=missing):
            generator.generate(Query(qid="0", query="What is the capital of Brazil?", **query_kwargs))

    def test_generate_experiment_skips_queries_that_already_have_a_rubric(self, llm_provider_mock, experiment):
        llm_provider_mock.async_call_mocker = AsyncMock(
            side_effect=lambda *args, **kwargs: LLMResponseType(
                raw_answer="rubric", parsed_answer=RubricSchema(criteria=_criteria())
            )
        )
        experiment["0"].rubric = [Criterion(criterion_name="kept", short_question="Untouched?")]
        generator = RubricGenerator(RubricGeneratorConfig(expert_in="geography"), llm_provider_mock)

        generator.generate_experiment(experiment, should_save=False)

        assert [c.criterion_name for c in experiment["0"].rubric] == ["kept"]
        assert [c.criterion_name for c in experiment["1"].rubric] == ["capital", "since_when"]
        assert llm_provider_mock.async_call_mocker.call_count == 1

    def test_get_rubric_generator_builds_a_config_from_kwargs(self, llm_provider_mock):
        generator = get_rubric_generator(llm_provider_mock, expert_in="geography", n_criteria=9)
        assert generator.config.expert_in == "geography"
        assert generator.config.n_criteria == 9


def _graded(query: Query, agent: str, fingerprint: str | None) -> AgentAnswer:
    answer = AgentAnswer(qid=query.qid, agent=agent, text=f"{agent} says Brasilia")
    answer.evaluations["rubric_pointwise"] = AnswerEvaluatorResult(
        qid=query.qid,
        agent=agent,
        evaluator_name="rubric_pointwise",
        answer=RubricPointwiseAnswerFormat(
            rubric_fingerprint=fingerprint,
            criteria=[
                CriterionEvaluationPointwise(
                    criterion=query.rubric[0], reasoning=f"{agent} reasoning", fulfillment=False
                ),
                CriterionEvaluationPointwise(
                    criterion=Criterion(criterion_name="evidence_recall", short_question="Recalled?"),
                    reasoning="1 of 3 snippets",
                    fulfillment=0.3,
                ),
            ],
        ),
    )
    query.answers[agent] = answer
    return answer


class TestRubricRefinement:
    def test_refine_shows_the_answers_graded_against_the_current_rubric(self, llm_provider_mock):
        _responds_with_criteria(llm_provider_mock)
        query = Query(
            qid="0",
            query="What is the capital of Brazil?",
            reference_answer="Brasilia, since 1960.",
            rubric=[Criterion(criterion_name="capital", short_question="Does it name the capital?", evidence=["d1"])],
            retrieved_docs={"d1": Document(qid="0", did="d1", text="Brasilia became the capital in 1960.")},
        )
        _graded(query, "agent1", query.rubric_fingerprint)
        _graded(query, "agent2", "an older rubric")

        rubric = RubricGenerator(RubricGeneratorConfig(n_criteria=3), llm_provider_mock).refine(query)

        assert [c.criterion_name for c in rubric] == ["capital", "since_when"]
        assert [c.criterion_name for c in query.rubric] == ["capital"]
        prompt = llm_provider_mock.async_call_mocker.call_args_list[0][0][0]
        assert "at most 3 criteria" in prompt.system_prompt
        assert "Brasilia, since 1960." in prompt.user_message
        assert "capital: Does it name the capital? Evidence: d1" in prompt.user_message
        assert "[[d1]] Brasilia became the capital in 1960." in prompt.user_message
        assert "evidence_recall" not in prompt.user_message
        assert "[[agent1]] agent1 says Brasilia" in prompt.user_message
        assert "- capital: False. agent1 reasoning" in prompt.user_message
        assert "[[agent2]] agent2 says Brasilia" in prompt.user_message
        assert "agent2 reasoning" not in prompt.user_message

    def test_refine_experiment_keeps_the_replaced_rubric(self, llm_provider_mock, experiment):
        _responds_with_criteria(llm_provider_mock)
        kept = [Criterion(criterion_name="kept", short_question="Untouched?")]
        experiment["0"].rubric = kept
        experiment["1"].rubric = []

        RubricGenerator(RubricGeneratorConfig(), llm_provider_mock).refine_experiment(experiment, should_save=False)

        assert [c.criterion_name for c in experiment["0"].rubric] == ["capital", "since_when"]
        assert experiment["0"].rubric_history == [kept]
        assert experiment["1"].rubric == [] and experiment["1"].rubric_history == []
        assert llm_provider_mock.async_call_mocker.call_count == 1

    def test_refine_needs_a_rubric(self, llm_provider_mock):
        with pytest.raises(ValueError, match="no rubric to refine"):
            RubricGenerator(RubricGeneratorConfig(), llm_provider_mock).refine(Query(qid="0", query="q"))


class TestRubricFingerprint:
    def test_fingerprint_tracks_the_criteria_and_ignores_their_order(self):
        query = Query(qid="0", query="q", rubric=_criteria())
        fingerprint = query.rubric_fingerprint

        query.rubric = list(reversed(query.rubric))
        assert query.rubric_fingerprint == fingerprint

        query.rubric[0].short_question = "Does it say since when, exactly?"
        assert query.rubric_fingerprint != fingerprint

    def test_a_query_without_a_rubric_has_no_fingerprint(self):
        assert Query(qid="0", query="q").rubric_fingerprint is None
