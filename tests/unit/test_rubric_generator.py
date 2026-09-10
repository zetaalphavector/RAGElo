from unittest.mock import AsyncMock

import pytest

from ragelo.generators import RubricGenerator, get_rubric_generator
from ragelo.types.answer_formats import Criterion, RubricSchema
from ragelo.types.configurations import RubricGeneratorConfig
from ragelo.types.evaluables import ChatMessage
from ragelo.types.formats import LLMResponseType
from ragelo.types.query import Query


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
