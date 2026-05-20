import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from tenacity import RetryError

from ragelo.types.configurations import LiteLLMConfiguration
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.results import PairwiseEvaluationAnswer, RetrievalEvaluationAnswer

# Stub litellm before importing the provider
_fake_litellm = types.ModuleType("litellm")
_fake_exceptions = types.ModuleType("litellm.exceptions")


class _FakeAuthenticationError(Exception):
    pass


class _FakeNotFoundError(Exception):
    pass


class _FakeRateLimitError(Exception):
    pass


class _FakeTimeout(Exception):
    pass


class _FakeBadRequestError(Exception):
    pass


_fake_exceptions.AuthenticationError = _FakeAuthenticationError
_fake_exceptions.NotFoundError = _FakeNotFoundError
_fake_exceptions.RateLimitError = _FakeRateLimitError
_fake_exceptions.Timeout = _FakeTimeout
_fake_exceptions.BadRequestError = _FakeBadRequestError
_fake_exceptions.APIConnectionError = type("APIConnectionError", (Exception,), {"__module__": "litellm.exceptions"})
_fake_exceptions.InternalServerError = type("InternalServerError", (Exception,), {"__module__": "litellm.exceptions"})
_fake_exceptions.ServiceUnavailableError = type("ServiceUnavailableError", (Exception,), {"__module__": "litellm.exceptions"})

_FakeRateLimitError.__module__ = "litellm.exceptions"
_FakeRateLimitError.__qualname__ = "RateLimitError"
_FakeTimeout.__module__ = "litellm.exceptions"
_FakeTimeout.__qualname__ = "Timeout"

_fake_litellm.exceptions = _fake_exceptions
_fake_litellm.acompletion = AsyncMock()
sys.modules["litellm"] = _fake_litellm
sys.modules["litellm.exceptions"] = _fake_exceptions


def _make_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.fixture(autouse=True)
def reset_litellm_mock():
    _fake_litellm.acompletion.reset_mock()
    _fake_litellm.acompletion.side_effect = None
    retrieval_json = json.dumps({"reasoning": "Highly relevant", "score": 2})
    _fake_litellm.acompletion.return_value = _make_response(retrieval_json)
    yield


@pytest.fixture
def litellm_config():
    return LiteLLMConfiguration(
        model="anthropic/claude-sonnet-4-6",
        api_key=SecretStr("sk-test-123"),
        temperature=0.1,
        max_tokens=4096,
    )


@pytest.fixture
def litellm_config_no_key():
    return LiteLLMConfiguration(model="openai/gpt-4o-mini")


@pytest.fixture
def litellm_provider(litellm_config):
    from ragelo.llm_providers.litellm_client import LiteLLMProvider
    return LiteLLMProvider(config=litellm_config)


@pytest.fixture
def litellm_provider_no_key(litellm_config_no_key):
    from ragelo.llm_providers.litellm_client import LiteLLMProvider
    return LiteLLMProvider(config=litellm_config_no_key)


class TestLiteLLMProvider:
    """Tests for LiteLLMProvider covering structured output, error handling, and edge cases."""

    def test_retrieval_evaluation(self, litellm_provider):
        result = litellm_provider(
            LLMInputPrompt(user_message="Evaluate this document."),
            response_schema=RetrievalEvaluationAnswer,
        )
        assert isinstance(result, LLMResponseType)
        assert isinstance(result.parsed_answer, RetrievalEvaluationAnswer)
        assert result.parsed_answer.reasoning == "Highly relevant"
        assert result.parsed_answer.score == 2

    def test_pairwise_evaluation(self, litellm_provider):
        pairwise_json = json.dumps({
            "answer_a_analysis": "A is good",
            "answer_b_analysis": "B is less detailed",
            "comparison_reasoning": "A is better",
            "winner": "A",
        })
        _fake_litellm.acompletion.return_value = _make_response(pairwise_json)

        result = litellm_provider(
            LLMInputPrompt(user_message="Compare these two answers."),
            response_schema=PairwiseEvaluationAnswer,
        )
        assert isinstance(result.parsed_answer, PairwiseEvaluationAnswer)
        assert result.parsed_answer.winner == "A"

    def test_drop_params_always_true(self, litellm_provider):
        litellm_provider(
            LLMInputPrompt(user_message="Test"),
            response_schema=RetrievalEvaluationAnswer,
        )
        call_kwargs = _fake_litellm.acompletion.call_args[1]
        assert call_kwargs["drop_params"] is True

    def test_model_forwarded(self, litellm_provider):
        litellm_provider(
            LLMInputPrompt(user_message="Test"),
            response_schema=RetrievalEvaluationAnswer,
        )
        call_kwargs = _fake_litellm.acompletion.call_args[1]
        assert call_kwargs["model"] == "anthropic/claude-sonnet-4-6"

    def test_provider_prefixed_model_string(self, litellm_provider):
        litellm_provider(
            LLMInputPrompt(user_message="Test"),
            response_schema=RetrievalEvaluationAnswer,
        )
        call_kwargs = _fake_litellm.acompletion.call_args[1]
        assert "/" in call_kwargs["model"]

    def test_api_key_forwarded(self, litellm_provider):
        litellm_provider(
            LLMInputPrompt(user_message="Test"),
            response_schema=RetrievalEvaluationAnswer,
        )
        call_kwargs = _fake_litellm.acompletion.call_args[1]
        assert call_kwargs["api_key"] == "sk-test-123"

    def test_api_key_omitted_when_none(self, litellm_provider_no_key):
        litellm_provider_no_key(
            LLMInputPrompt(user_message="Test"),
            response_schema=RetrievalEvaluationAnswer,
        )
        call_kwargs = _fake_litellm.acompletion.call_args[1]
        assert "api_key" not in call_kwargs

    def test_temperature_forwarded(self, litellm_provider):
        litellm_provider(
            LLMInputPrompt(user_message="Test"),
            response_schema=RetrievalEvaluationAnswer,
        )
        call_kwargs = _fake_litellm.acompletion.call_args[1]
        assert call_kwargs["temperature"] == 0.1

    def test_system_and_user_prompt(self, litellm_provider):
        litellm_provider(
            LLMInputPrompt(system_prompt="You are an evaluator.", user_message="Evaluate this."),
            response_schema=RetrievalEvaluationAnswer,
        )
        call_kwargs = _fake_litellm.acompletion.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "You are an evaluator."}
        assert "Evaluate this." in messages[1]["content"]

    def test_messages_list_input(self, litellm_provider):
        messages = [
            {"role": "system", "content": "You are an evaluator."},
            {"role": "user", "content": "Evaluate this."},
        ]
        litellm_provider(
            LLMInputPrompt(messages=messages),
            response_schema=RetrievalEvaluationAnswer,
        )
        call_kwargs = _fake_litellm.acompletion.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "system"

    def test_json_schema_appended_to_prompt(self, litellm_provider):
        litellm_provider(
            LLMInputPrompt(user_message="Evaluate this."),
            response_schema=RetrievalEvaluationAnswer,
        )
        call_kwargs = _fake_litellm.acompletion.call_args[1]
        last_msg = call_kwargs["messages"][-1]["content"]
        assert "schema" in last_msg.lower()
        assert "JSON" in last_msg

    def test_response_format_json_object(self, litellm_provider):
        litellm_provider(
            LLMInputPrompt(user_message="Test"),
            response_schema=RetrievalEvaluationAnswer,
        )
        call_kwargs = _fake_litellm.acompletion.call_args[1]
        assert call_kwargs["response_format"] == {"type": "json_object"}


class TestLiteLLMProviderErrors:
    """Tests for error handling with litellm-specific exceptions."""

    def test_auth_error_not_retried(self, litellm_provider):
        _fake_litellm.acompletion.side_effect = _FakeAuthenticationError("Invalid API key")
        with pytest.raises(_FakeAuthenticationError, match="Invalid API key"):
            litellm_provider(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )
        assert _fake_litellm.acompletion.call_count == 1

    def test_not_found_error_not_retried(self, litellm_provider):
        _fake_litellm.acompletion.side_effect = _FakeNotFoundError("Model not found")
        with pytest.raises(_FakeNotFoundError, match="Model not found"):
            litellm_provider(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )
        assert _fake_litellm.acompletion.call_count == 1

    def test_bad_request_error_not_retried(self, litellm_provider):
        _fake_litellm.acompletion.side_effect = _FakeBadRequestError("context_length_exceeded")
        with pytest.raises(_FakeBadRequestError):
            litellm_provider(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )
        assert _fake_litellm.acompletion.call_count == 1

    def test_rate_limit_retried(self, litellm_provider):
        _fake_litellm.acompletion.side_effect = _fake_exceptions.RateLimitError("429")
        with pytest.raises(_fake_exceptions.RateLimitError):
            litellm_provider(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )
        assert _fake_litellm.acompletion.call_count == 3

    def test_timeout_retried(self, litellm_provider):
        _fake_litellm.acompletion.side_effect = _fake_exceptions.Timeout("timed out")
        with pytest.raises(_fake_exceptions.Timeout):
            litellm_provider(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )
        assert _fake_litellm.acompletion.call_count == 3

    def test_empty_response_raises(self, litellm_provider):
        empty_resp = MagicMock()
        empty_resp.choices = []
        _fake_litellm.acompletion.return_value = empty_resp
        with pytest.raises((RetryError, ValueError)):
            litellm_provider(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )

    def test_null_content_raises(self, litellm_provider):
        _fake_litellm.acompletion.return_value = _make_response(None)
        _fake_litellm.acompletion.return_value.choices[0].message.content = None
        with pytest.raises((RetryError, ValueError)):
            litellm_provider(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )

    def test_invalid_json_raises(self, litellm_provider):
        _fake_litellm.acompletion.return_value = _make_response("not valid json")
        with pytest.raises((RetryError, ValueError)):
            litellm_provider(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )

    def test_missing_required_field_raises(self, litellm_provider):
        _fake_litellm.acompletion.return_value = _make_response(json.dumps({"reasoning": "ok"}))
        with pytest.raises((RetryError, ValueError)):
            litellm_provider(
                LLMInputPrompt(user_message="Test"),
                response_schema=RetrievalEvaluationAnswer,
            )


class TestLiteLLMProviderFactory:
    """Tests for factory registration and creation."""

    def test_factory_creates_litellm_provider(self):
        from ragelo.llm_providers import get_llm_provider
        from ragelo.llm_providers.litellm_client import LiteLLMProvider

        provider = get_llm_provider("litellm", model="openai/gpt-4o-mini")
        assert isinstance(provider, LiteLLMProvider)
        assert provider.config.model == "openai/gpt-4o-mini"

    def test_factory_with_api_key(self):
        from ragelo.llm_providers import get_llm_provider

        provider = get_llm_provider("litellm", model="anthropic/claude-haiku-4-5", api_key="sk-test")
        assert provider.config.api_key.get_secret_value() == "sk-test"

    def test_factory_with_api_base(self):
        from ragelo.llm_providers import get_llm_provider

        provider = get_llm_provider("litellm", model="openai/gpt-4o", api_base="http://localhost:4000")
        assert provider.config.api_base == "http://localhost:4000"
