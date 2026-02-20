import json
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr, ValidationError
from tenacity import RetryError

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.llm_providers.openai_client import OpenAIProvider
from ragelo.types.configurations import LLMProviderConfig, OpenAIConfiguration
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.results import PairwiseEvaluationAnswer, RetrievalEvaluationAnswer


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
        assert call_args[1]["instructions"] is None

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
        assert "text" in call_args[1]
        assert call_args[1]["text"]["format"]["type"] == "json_object"

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

    def test_pairwise_evaluation_structured_mode(self, openai_provider_structured, flexible_openai_client_mock):
        """Test pairwise evaluation with structured output."""
        # Execute
        user_prompt = "Compare these two answers and determine which is better."
        result = openai_provider_structured(
            LLMInputPrompt(user_message=user_prompt),
            response_schema=PairwiseEvaluationAnswer,
        )

        # Assert
        assert isinstance(result, LLMResponseType)
        assert isinstance(result.parsed_answer, PairwiseEvaluationAnswer)
        assert result.parsed_answer.answer_a_analysis == "Answer A is comprehensive and accurate"
        assert result.parsed_answer.answer_b_analysis == "Answer B is less detailed"
        assert result.parsed_answer.comparison_reasoning == "Answer A provides more depth"
        assert result.parsed_answer.winner == "A"

        # Verify responses.parse was called
        assert flexible_openai_client_mock.responses.parse.called
        call_args = flexible_openai_client_mock.responses.parse.call_args
        assert call_args[1]["text_format"] == PairwiseEvaluationAnswer

    def test_pairwise_evaluation_json_mode(self, openai_provider_json_mode, flexible_openai_client_mock):
        """Test pairwise evaluation with JSON mode."""

        # Update mock to return pairwise data for json_mode
        def create_side_effect_pairwise(*args, **kwargs):
            resp = flexible_openai_client_mock.responses.create.return_value
            resp.output_text = json.dumps(
                {
                    "answer_a_analysis": "Answer A is comprehensive and accurate",
                    "answer_b_analysis": "Answer B is less detailed",
                    "comparison_reasoning": "Answer A provides more depth",
                    "winner": "A",
                }
            )
            return resp

        flexible_openai_client_mock.responses.create.side_effect = create_side_effect_pairwise

        # Execute
        user_prompt = "Compare these two answers."
        result = openai_provider_json_mode(
            LLMInputPrompt(user_message=user_prompt),
            response_schema=PairwiseEvaluationAnswer,
        )

        # Assert
        assert isinstance(result, LLMResponseType)
        assert isinstance(result.parsed_answer, PairwiseEvaluationAnswer)
        assert result.parsed_answer.winner == "A"

        # Verify responses.create was called
        assert flexible_openai_client_mock.responses.create.called

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
        """Test that invalid JSON in json_mode raises a RetryError (wrapping ValueError)."""

        # Mock to return invalid JSON
        def create_invalid_json(*args, **kwargs):
            resp = flexible_openai_client_mock.responses.create.return_value
            resp.output_text = "This is not valid JSON"
            return resp

        flexible_openai_client_mock.responses.create.side_effect = create_invalid_json

        # Execute & Assert - the retry decorator wraps ValueError in RetryError
        with pytest.raises(RetryError) as exc_info:
            openai_provider_json_mode(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )

        # Check that the underlying exception was a ValueError about JSON parsing
        assert exc_info.value.last_attempt.exception() is not None
        underlying_error = str(exc_info.value.last_attempt.exception())
        assert "Failed to parse raw JSON answer" in underlying_error

    def test_missing_required_field_in_json_mode_raises_error(
        self, openai_provider_json_mode, flexible_openai_client_mock
    ):
        """Test that JSON missing required fields raises a RetryError (wrapping ValueError)."""

        # Mock to return JSON missing required fields
        def create_incomplete_json(*args, **kwargs):
            resp = flexible_openai_client_mock.responses.create.return_value
            resp.output_text = json.dumps({"reasoning": "Only reasoning, no score"})
            return resp

        flexible_openai_client_mock.responses.create.side_effect = create_incomplete_json

        # Execute & Assert - the retry decorator wraps ValueError in RetryError
        with pytest.raises(RetryError) as exc_info:
            openai_provider_json_mode(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )

        # Check that the underlying exception was a ValueError about missing field
        assert exc_info.value.last_attempt.exception() is not None
        underlying_error = str(exc_info.value.last_attempt.exception())
        assert "Failed to parse raw JSON answer" in underlying_error

    def test_wrong_type_from_structured_mode_raises_error(
        self, openai_provider_structured, flexible_openai_client_mock
    ):
        """Test that wrong type from structured mode raises a RetryError (wrapping ValueError)."""

        # Mock to return wrong type
        def parse_wrong_type(*args, **kwargs):
            resp = flexible_openai_client_mock.responses.parse.return_value
            resp.output_text = '{"wrong": "type"}'
            resp.output_parsed = "Not the right type"  # String instead of RetrievalEvaluationAnswer
            return resp

        flexible_openai_client_mock.responses.parse.side_effect = parse_wrong_type

        # Execute & Assert - the retry decorator wraps ValueError in RetryError
        with pytest.raises(RetryError) as exc_info:
            openai_provider_structured(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )

        # Check that the underlying exception was a ValueError about type mismatch
        assert exc_info.value.last_attempt.exception() is not None
        underlying_error = str(exc_info.value.last_attempt.exception())
        assert "OpenAI failed to parse response" in underlying_error

    def test_temperature_set_to_none_for_reasoning_models(self, monkeypatch):
        """Test that temperature is set to None for reasoning models (gpt-5, o-series)."""
        # Test with gpt-5 model
        config_gpt5 = OpenAIConfiguration(
            api_key=SecretStr("fake_key"),
            model="gpt-5-preview",
            temperature=0.7,
        )
        provider_gpt5 = OpenAIProvider(config=config_gpt5)
        assert provider_gpt5.config is not None
        assert provider_gpt5.config.temperature is None

        # Test with o-series model
        config_o = OpenAIConfiguration(
            api_key=SecretStr("fake_key"),
            model="o1-preview",
            temperature=0.7,
        )
        provider_o = OpenAIProvider(config=config_o)
        assert provider_o.config is not None
        assert provider_o.config.temperature is None

        # Test with regular model keeps temperature
        config_regular = OpenAIConfiguration(
            api_key=SecretStr("fake_key"),
            model="gpt-4o-mini",
            temperature=0.7,
        )
        provider_regular = OpenAIProvider(config=config_regular)
        assert provider_regular.config
        assert provider_regular.config.temperature == 0.7


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

    def test_base_llm_provider_still_accepts_config(self):
        class SimpleProvider(BaseLLMProvider):
            async def call_async(self, input, response_schema): ...

        config = LLMProviderConfig()
        provider = SimpleProvider(config=config)
        assert provider.config is config

    def test_get_config_class_resolves_union_annotation(self):
        """get_config_class() returns the concrete LLMProviderConfig subclass even when
        the class attribute is annotated as 'LLMProviderConfig | None'."""

        assert BaseLLMProvider.get_config_class() is LLMProviderConfig

    def test_external_adapter_call_delegates_to_call_async(self):
        """External adapter's __call__ method correctly delegates to call_async."""

        expected = LLMResponseType(
            raw_answer='{"reasoning": "ok", "score": 1}',
            parsed_answer=RetrievalEvaluationAnswer(reasoning="ok", score=1),
        )

        class ExternalAdapterProvider(BaseLLMProvider):
            def __init__(self, client):
                super().__init__()
                self.client = client

            async def call_async(self, input, response_schema): ...

        provider = ExternalAdapterProvider(client=object())
        provider.call_async = AsyncMock(return_value=expected)  # type: ignore[method-assign]
        result = provider(LLMInputPrompt(user_message="test"), RetrievalEvaluationAnswer)
        assert result == expected


class TestLLMProviderConfigOptionalApiKey:
    """Tests that api_key is optional in the base config but required in OpenAIConfiguration."""

    def test_llm_provider_config_can_be_created_without_api_key(self):
        """LLMProviderConfig can now be instantiated without providing api_key."""

        config = LLMProviderConfig()
        assert config.api_key is None

    def test_llm_provider_config_accepts_api_key_when_provided(self):
        """LLMProviderConfig still accepts api_key when supplied."""

        config = LLMProviderConfig(api_key=SecretStr("my-secret"))
        assert config.api_key is not None
        assert config.api_key.get_secret_value() == "my-secret"

    def test_openai_configuration_requires_api_key(self):
        """OpenAIConfiguration.api_key is still required (no default)."""

        with pytest.raises(ValidationError):
            OpenAIConfiguration()  # missing api_key
