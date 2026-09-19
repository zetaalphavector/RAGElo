import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr, ValidationError

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.llm_providers.openai_client import OpenAIProvider
from ragelo.types.configurations import LLMProviderConfig, OpenAIConfiguration
from ragelo.types.formats import LLMInputPrompt, LLMResponseType, LLMUsage
from ragelo.types.results import RetrievalEvaluationAnswer


class TestOpenAIProvider:
    """Tests for OpenAIProvider with real-world answer schemas.

    These tests cover:
    - Both json_mode=True and json_mode=False configurations
    - Real answer schemas (RetrievalEvaluationAnswer, PairwiseEvaluationAnswer)
    - Different input formats (user message only, system + user, messages list)
    - Error handling scenarios
    """

    def test_retrieval_evaluation_structured_mode(self, openai_provider_structured, flexible_openai_client_mock):
        """Test retrieval evaluation with json_mode=False (structured output via responses.parse)."""
        # Execute
        user_prompt = "Evaluate the relevance of this document to the query."
        result = openai_provider_structured(
            LLMInputPrompt(user_message=user_prompt),
            response_schema=RetrievalEvaluationAnswer,
        )

        # Assert - verify correct response type
        assert isinstance(result, LLMResponseType)
        assert isinstance(result.parsed_answer, RetrievalEvaluationAnswer)
        assert result.parsed_answer.reasoning == "The document is highly relevant to the query"
        assert result.parsed_answer.score == 2
        assert result.raw_answer == json.dumps(
            {"reasoning": "The document is highly relevant to the query", "score": 2}
        )

        # Assert - verify responses.parse was called (not responses.create)
        assert flexible_openai_client_mock.responses.parse.called
        assert not flexible_openai_client_mock.responses.create.called

        # Assert - verify correct parameters were passed
        call_args = flexible_openai_client_mock.responses.parse.call_args
        assert call_args[1]["model"] == "fake_model"
        assert call_args[1]["input"] == user_prompt
        assert call_args[1]["text_format"] == RetrievalEvaluationAnswer
        assert "instructions" not in call_args[1]

    @pytest.mark.parametrize("provider_fixture", ["openai_provider_structured", "openai_provider_json_mode"])
    def test_the_billed_tokens_are_reported_with_the_cached_part_of_the_input(self, provider_fixture, request):
        provider = request.getfixturevalue(provider_fixture)

        result = provider(LLMInputPrompt(user_message="Evaluate."), response_schema=RetrievalEvaluationAnswer)

        assert result.usage == LLMUsage(input_tokens=120, output_tokens=30, cached_tokens=20)

    def test_retrieval_evaluation_json_mode(self, openai_provider_json_mode, flexible_openai_client_mock):
        """Test retrieval evaluation with json_mode=True (JSON string via responses.create)."""
        # Execute
        user_prompt = "Evaluate the relevance of this document to the query."
        result = openai_provider_json_mode(
            LLMInputPrompt(user_message=user_prompt),
            response_schema=RetrievalEvaluationAnswer,
        )

        # Assert - verify correct response type
        assert isinstance(result, LLMResponseType)
        assert isinstance(result.parsed_answer, RetrievalEvaluationAnswer)
        assert result.parsed_answer.reasoning == "The document is highly relevant to the query"
        assert result.parsed_answer.score == 2

        # Assert - verify responses.create was called (not responses.parse)
        assert flexible_openai_client_mock.responses.create.called
        assert not flexible_openai_client_mock.responses.parse.called

        # Assert - verify schema was appended to the user prompt
        call_args = flexible_openai_client_mock.responses.create.call_args
        assert call_args[1]["model"] == "fake_model"

        # Verify the schema is appended to the input
        input_arg = call_args[1]["input"]
        assert user_prompt in input_arg
        assert "schema" in input_arg.lower()  # Schema description should be in the prompt

        # Verify JSON mode response format is set
        assert call_args[1]["text"]["format"]["type"] == "json_schema"
        assert call_args[1]["text"]["format"]["strict"] is False

    def test_retrieval_evaluation_with_system_prompt_structured(
        self, openai_provider_structured, flexible_openai_client_mock
    ):
        """Test retrieval evaluation with both system and user prompts in structured mode."""
        # Execute
        system_prompt = "You are a relevance evaluation expert."
        user_prompt = "Evaluate this document."
        result = openai_provider_structured(
            LLMInputPrompt(system_prompt=system_prompt, user_message=user_prompt),
            response_schema=RetrievalEvaluationAnswer,
        )

        # Assert
        assert isinstance(result, LLMResponseType)
        assert isinstance(result.parsed_answer, RetrievalEvaluationAnswer)

        # Verify system prompt was passed correctly
        call_args = flexible_openai_client_mock.responses.parse.call_args
        assert call_args[1]["instructions"] == system_prompt
        assert call_args[1]["input"] == user_prompt

    def test_retrieval_evaluation_with_system_prompt_json_mode(
        self, openai_provider_json_mode, flexible_openai_client_mock
    ):
        """Test retrieval evaluation with both system and user prompts in JSON mode."""
        # Execute
        system_prompt = "You are a relevance evaluation expert."
        user_prompt = "Evaluate this document."
        result = openai_provider_json_mode(
            LLMInputPrompt(system_prompt=system_prompt, user_message=user_prompt),
            response_schema=RetrievalEvaluationAnswer,
        )

        # Assert
        assert isinstance(result, LLMResponseType)

        # Verify system prompt and user prompt were passed correctly
        call_args = flexible_openai_client_mock.responses.create.call_args
        assert call_args[1]["instructions"] == system_prompt

        # Verify the schema was appended to user message
        input_arg = call_args[1]["input"]
        assert user_prompt in input_arg
        assert "schema" in input_arg.lower()

    def test_messages_list_input(self, openai_provider_structured, flexible_openai_client_mock):
        """Test provider with messages list input format."""
        # Execute
        messages = [
            {"role": "system", "content": "You are an evaluator."},
            {"role": "user", "content": "Evaluate this."},
        ]
        result = openai_provider_structured(
            LLMInputPrompt(messages=messages),
            response_schema=RetrievalEvaluationAnswer,
        )

        # Assert
        assert isinstance(result, LLMResponseType)

        # Verify messages were passed correctly
        call_args = flexible_openai_client_mock.responses.parse.call_args
        assert call_args[1]["input"] == messages

    def test_invalid_json_in_json_mode_raises_error(self, openai_provider_json_mode, flexible_openai_client_mock):
        """Test that invalid JSON in json_mode raises the parse ValueError."""

        # Mock to return invalid JSON
        def create_invalid_json(*args, **kwargs):
            resp = flexible_openai_client_mock.responses.create.return_value
            resp.output_text = "This is not valid JSON"
            return resp

        flexible_openai_client_mock.responses.create.side_effect = create_invalid_json

        # Execute & Assert
        with pytest.raises(ValueError) as exc_info:
            openai_provider_json_mode(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )

        assert "Failed to parse raw JSON answer" in str(exc_info.value)

    def test_missing_required_field_in_json_mode_raises_error(
        self, openai_provider_json_mode, flexible_openai_client_mock
    ):
        """Test that JSON missing required fields raises the parse ValueError."""

        # Mock to return JSON missing required fields
        def create_incomplete_json(*args, **kwargs):
            resp = flexible_openai_client_mock.responses.create.return_value
            resp.output_text = json.dumps({"reasoning": "Only reasoning, no score"})
            return resp

        flexible_openai_client_mock.responses.create.side_effect = create_incomplete_json

        # Execute & Assert
        with pytest.raises(ValueError) as exc_info:
            openai_provider_json_mode(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )

        assert "Failed to parse raw JSON answer" in str(exc_info.value)

    def test_wrong_type_from_structured_mode_raises_error(
        self, openai_provider_structured, flexible_openai_client_mock
    ):
        """Test that wrong type from structured mode raises the parse ValueError."""

        # Mock to return wrong type
        def parse_wrong_type(*args, **kwargs):
            resp = flexible_openai_client_mock.responses.parse.return_value
            resp.output_text = '{"wrong": "type"}'
            resp.output_parsed = "Not the right type"  # String instead of RetrievalEvaluationAnswer
            return resp

        flexible_openai_client_mock.responses.parse.side_effect = parse_wrong_type

        # Execute & Assert
        with pytest.raises(ValueError) as exc_info:
            openai_provider_structured(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )

        assert "OpenAI failed to parse response" in str(exc_info.value)

    def test_json_mode_leaves_the_prompt_messages_untouched(self, openai_provider_json_mode):
        prompt = LLMInputPrompt(messages=[{"role": "user", "content": "Evaluate this document."}])
        openai_provider_json_mode(prompt, response_schema=RetrievalEvaluationAnswer)
        assert prompt.messages == [{"role": "user", "content": "Evaluate this document."}]

    def test_failed_request_is_attempted_once(self, openai_provider_structured, flexible_openai_client_mock):
        flexible_openai_client_mock.responses.parse = AsyncMock(side_effect=RuntimeError("boom"))
        with pytest.raises(ValueError, match="boom"):
            openai_provider_structured(LLMInputPrompt(user_message="Test"), response_schema=RetrievalEvaluationAnswer)
        assert flexible_openai_client_mock.responses.parse.await_count == 1

    def test_configured_sampling_params_are_sent_as_is(self, flexible_openai_client_mock):
        config = OpenAIConfiguration(
            api_key=SecretStr("fake_key"), model="gpt-5.6-luna", temperature=0.7, reasoning_effort="high"
        )
        provider = OpenAIProvider(config=config, client=flexible_openai_client_mock)
        provider(LLMInputPrompt(user_message="Test"), response_schema=RetrievalEvaluationAnswer)
        call_kwargs = flexible_openai_client_mock.responses.parse.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["reasoning"] == {"effort": "high"}

    def test_unset_sampling_params_are_omitted(self, openai_provider_structured, flexible_openai_client_mock):
        openai_provider_structured(LLMInputPrompt(user_message="Test"), response_schema=RetrievalEvaluationAnswer)
        call_kwargs = flexible_openai_client_mock.responses.parse.call_args.kwargs
        assert "temperature" not in call_kwargs
        assert "reasoning" not in call_kwargs


class TestExternalAdapterProvider:
    def test_external_adapter_can_be_instantiated_without_config(self):
        """Simulates an external adapter that wraps a pre-configured client."""

        class ExternalAdapterProvider(BaseLLMProvider):
            def __init__(self, some_client):
                super().__init__()
                self.client = some_client

            async def call_async(self, input, response_schema): ...

        provider = ExternalAdapterProvider(some_client=object())
        assert provider.config is None

    def test_get_config_class_resolves_union_annotation(self):
        """get_config_class() returns the concrete LLMProviderConfig subclass even when
        the class attribute is annotated as 'LLMProviderConfig | None'."""

        assert BaseLLMProvider.get_config_class() is LLMProviderConfig


class TestLLMProviderFactoryArguments:
    def test_unknown_argument_is_rejected(self):
        from ragelo.llm_providers import get_llm_provider

        with pytest.raises(ValidationError, match="temprature"):
            get_llm_provider("openai", api_key="fake_key", temprature=0)

    def test_evaluator_factory_routes_shared_arguments(self):
        from ragelo import get_retrieval_evaluator

        evaluator = get_retrieval_evaluator(
            "reasoner", llm_provider="openai", api_key="fake_key", model="fake-model", n_processes=3
        )
        assert evaluator.llm_provider.config.model == "fake-model"
        assert evaluator.config.n_processes == 3


class TestOllamaProviderFactory:
    """Tests for creating OllamaProvider through the factory."""

    def test_get_llm_provider_ollama_without_api_key_env(self, monkeypatch):
        """get_llm_provider('ollama', model=...) works even when OPENAI_API_KEY is not set."""
        from ragelo.llm_providers import get_llm_provider
        from ragelo.llm_providers.ollama_client import OllamaProvider

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = get_llm_provider("ollama", model="test-model")
        assert isinstance(provider, OllamaProvider)
        assert provider.config.model == "test-model"


class TestOllamaProvider:
    def test_json_mode_parses_a_valid_answer(self):
        from ragelo.llm_providers import OllamaProvider
        from ragelo.types.configurations import OllamaConfiguration

        raw_answer = json.dumps({"reasoning": "Relevant", "score": 2})
        client = MagicMock()
        client.chat.completions.create = AsyncMock(
            return_value=MagicMock(choices=[MagicMock(message=MagicMock(content=raw_answer))])
        )
        provider = OllamaProvider(OllamaConfiguration(model="test-model", json_mode=True), client=client)
        prompt = LLMInputPrompt(messages=[{"role": "user", "content": "Evaluate this document."}])

        result = provider(prompt, response_schema=RetrievalEvaluationAnswer)

        assert result.parsed_answer == RetrievalEvaluationAnswer(reasoning="Relevant", score=2)
        assert result.raw_answer == raw_answer
        assert prompt.messages == [{"role": "user", "content": "Evaluate this document."}]
        sent = client.chat.completions.create.call_args.kwargs["messages"]
        assert "STRICTLY adheres" in sent[-1]["content"]


class TestUsage:
    def test_openai_usage_without_token_details_counts_no_cached_tokens(
        self, openai_provider_structured, flexible_openai_client_mock
    ):
        parse = flexible_openai_client_mock.responses.parse.side_effect

        def without_details(*args, **kwargs):
            response = parse(*args, **kwargs)
            response.usage = MagicMock(input_tokens=120, output_tokens=30, input_tokens_details=None)
            return response

        flexible_openai_client_mock.responses.parse.side_effect = without_details
        result = openai_provider_structured(
            LLMInputPrompt(user_message="Test"), response_schema=RetrievalEvaluationAnswer
        )
        assert result.usage == LLMUsage(input_tokens=120, output_tokens=30, cached_tokens=0)

    def test_ollama_reports_the_prompt_and_completion_tokens(self):
        from ragelo.llm_providers import OllamaProvider
        from ragelo.types.configurations import OllamaConfiguration

        answer = RetrievalEvaluationAnswer(reasoning="Relevant", score=2)
        message = MagicMock(content=answer.model_dump_json(), parsed=answer)
        client = MagicMock()
        client.chat.completions.parse = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=message)], usage=MagicMock(prompt_tokens=90, completion_tokens=12)
            )
        )
        provider = OllamaProvider(OllamaConfiguration(model="test-model"), client=client)

        result = provider(LLMInputPrompt(user_message="Test"), response_schema=RetrievalEvaluationAnswer)

        assert result.usage == LLMUsage(input_tokens=90, output_tokens=12)

    @pytest.mark.parametrize(
        "raw_usage",
        [
            MagicMock(spec=["prompt_tokens", "completion_tokens"], prompt_tokens=90, completion_tokens=12),
            MagicMock(spec=["input_tokens", "output_tokens"], input_tokens=90, output_tokens=12),
        ],
        ids=["openai style", "anthropic style"],
    )
    def test_instructor_reads_the_usage_either_sdk_style_reports(
        self, instructor_provider, instructor_client_mock, raw_usage
    ):
        answer = RetrievalEvaluationAnswer(reasoning="Relevant", score=2)
        object.__setattr__(answer, "_raw_response", MagicMock(usage=raw_usage))
        instructor_client_mock.create = AsyncMock(return_value=answer)

        result = instructor_provider(LLMInputPrompt(user_message="Test"), response_schema=RetrievalEvaluationAnswer)

        assert result.usage == LLMUsage(input_tokens=90, output_tokens=12)


class TestVercelProvider:
    """Tests for the Vercel AI Gateway provider."""

    def test_factory_reads_the_gateway_key(self, monkeypatch):
        from ragelo.llm_providers import VercelProvider, get_llm_provider

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AI_GATEWAY_API_KEY", "gateway-key")
        provider = get_llm_provider("vercel", model="anthropic/claude-sonnet-5")
        assert isinstance(provider, VercelProvider)
        assert provider.config.api_key.get_secret_value() == "gateway-key"
        assert provider.config.api_base == "https://ai-gateway.vercel.sh/v1"

    def test_factory_ignores_the_openai_key(self, monkeypatch):
        from ragelo.llm_providers import get_llm_provider

        monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
        monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
        with pytest.raises(ValueError, match="AI_GATEWAY_API_KEY"):
            get_llm_provider("vercel", model="anthropic/claude-sonnet-5")


class TestInstructorProvider:
    """Tests for InstructorProvider with instructor-patched clients.

    These tests cover:
    - Both chat.completions.create (OpenAI/Mistral/Cohere) and messages.create (Anthropic) paths
    - Real answer schemas (RetrievalEvaluationAnswer, PairwiseEvaluationAnswer)
    - Different input formats (user message only, system + user, messages list)
    - Error handling
    """

    def test_retrieval_evaluation(self, instructor_provider, instructor_client_mock):
        """Test retrieval evaluation returns correct LLMResponseType."""
        pytest.importorskip("instructor")
        user_prompt = "Evaluate the relevance of this document to the query."
        result = instructor_provider(
            LLMInputPrompt(user_message=user_prompt),
            response_schema=RetrievalEvaluationAnswer,
        )

        assert isinstance(result, LLMResponseType)
        assert isinstance(result.parsed_answer, RetrievalEvaluationAnswer)
        assert result.parsed_answer.reasoning == "The document is highly relevant to the query"
        assert result.parsed_answer.score == 2
        assert instructor_client_mock.create.called
        call_args = instructor_client_mock.create.call_args
        assert call_args[1]["response_model"] == RetrievalEvaluationAnswer

    def test_system_and_user_prompt(self, instructor_provider, instructor_client_mock):
        """Test that system and user prompts are built into the messages list correctly."""
        pytest.importorskip("instructor")
        instructor_provider(
            LLMInputPrompt(system_prompt="You are an evaluator.", user_message="Evaluate this."),
            response_schema=RetrievalEvaluationAnswer,
        )

        call_args = instructor_client_mock.create.call_args
        messages = call_args[1]["messages"]
        assert messages[0] == {"role": "system", "content": "You are an evaluator."}
        assert messages[1] == {"role": "user", "content": "Evaluate this."}

    def test_messages_list_input(self, instructor_provider, instructor_client_mock):
        """Test that a pre-built messages list is passed through unchanged."""
        pytest.importorskip("instructor")
        messages = [
            {"role": "system", "content": "You are an evaluator."},
            {"role": "user", "content": "Evaluate this."},
        ]
        instructor_provider(
            LLMInputPrompt(messages=messages),
            response_schema=RetrievalEvaluationAnswer,
        )

        call_args = instructor_client_mock.create.call_args
        assert call_args[1]["messages"] == messages

    def test_call_leaves_the_configured_model_kwargs_untouched(self, instructor_provider):
        instructor_provider(LLMInputPrompt(user_message="Test"), response_schema=RetrievalEvaluationAnswer)
        assert instructor_provider.config.model_kwargs == {}

    def test_api_error_reraises_the_underlying_error(self, instructor_provider, instructor_client_mock):
        """Test that an API failure raises the underlying ValueError."""
        pytest.importorskip("instructor")
        instructor_client_mock.create.side_effect = RuntimeError("Connection refused")

        with pytest.raises(ValueError) as exc_info:
            instructor_provider(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )

        assert "Instructor request failed" in str(exc_info.value)
        assert "Connection refused" in str(exc_info.value)

    def test_unknown_provider_raises_value_error(self):
        """Test that an unrecognized provider/model string raises ValueError at instantiation."""
        pytest.importorskip("instructor")
        from ragelo.llm_providers.instructor_client import InstructorProvider
        from ragelo.types.configurations.llm_provider_configs import InstructorConfiguration

        config = InstructorConfiguration(model="unsupported_xyz/some-model")
        with pytest.raises(ValueError, match="Failed to initialize instructor client"):
            InstructorProvider(config=config)

    def test_get_llm_provider_instructor_via_factory(self, monkeypatch):
        """Test that get_llm_provider('instructor', ...) returns InstructorProvider."""
        pytest.importorskip("instructor")
        import instructor

        from ragelo.llm_providers import get_llm_provider
        from ragelo.llm_providers.instructor_client import InstructorProvider

        mock_client = MagicMock()
        mock_client.create = AsyncMock(return_value=RetrievalEvaluationAnswer(reasoning="ok", score=1))
        monkeypatch.setattr(instructor, "from_provider", lambda *args, **kwargs: mock_client)
        provider = get_llm_provider("instructor", model="openai/fake-model")
        assert isinstance(provider, InstructorProvider)
        assert provider.config.model == "openai/fake-model"

    def test_api_key_forwarded_to_from_provider(self, monkeypatch):
        """Test that an explicit api_key is forwarded to instructor.from_provider."""
        pytest.importorskip("instructor")
        import instructor

        from ragelo.llm_providers import get_llm_provider

        captured_kwargs: dict = {}

        def spy_from_provider(*args, **kwargs):
            captured_kwargs.update(kwargs)
            mock_client = MagicMock()
            mock_client.create = AsyncMock(return_value=RetrievalEvaluationAnswer(reasoning="ok", score=1))
            return mock_client

        monkeypatch.setattr(instructor, "from_provider", spy_from_provider)
        get_llm_provider("instructor", model="anthropic/fake-model", api_key="test-secret-key")
        assert captured_kwargs.get("api_key") == "test-secret-key"

    @pytest.mark.requires_anthropic
    def test_anthropic_integration(self):
        """Integration test: real Anthropic call via InstructorProvider.

        Requires: ``pip install anthropic`` and ANTHROPIC_API_KEY env var.
        Run with: pytest --runanthropic
        """
        pytest.importorskip("anthropic")
        from ragelo.llm_providers import get_llm_provider

        provider = get_llm_provider(
            "instructor", model="anthropic/claude-haiku-4-5", api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
        result = provider(
            LLMInputPrompt(
                user_message=("Is Paris the capital of France? Provide a short reasoning and score 2 if yes, 0 if no.")
            ),
            response_schema=RetrievalEvaluationAnswer,
        )
        assert isinstance(result, LLMResponseType)
        assert isinstance(result.parsed_answer, RetrievalEvaluationAnswer)
        assert result.parsed_answer.score == 2
