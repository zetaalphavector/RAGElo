from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel as PydanticBaseModel

from ragelo.llm_providers.base_llm_provider import get_llm_provider
from ragelo.llm_providers.openai_client import OpenAIProvider
from ragelo.types.formats import AnswerFormat, LLMResponseType


class AnswerModel(PydanticBaseModel):
    keyA: str
    keyB: str


class TestOpenAIProvider:
    @pytest.mark.parametrize(
        "answer_format,response_schema,raw_answer,parsed_answer",
        [
            (AnswerFormat.TEXT, None, "fake response", "fake response"),
            (
                AnswerFormat.JSON,
                {"keyA": "Value of key a", "KeyB": "Value of key b"},
                '{"keyA": "valueA", "keyB": "valueB"}',
                {"keyA": "valueA", "keyB": "valueB"},
            ),
            (
                AnswerFormat.STRUCTURED,
                AnswerModel,
                '{"keyA": "valueA", "keyB": "valueB"}',
                AnswerModel(keyA="valueA", keyB="valueB"),
            ),
        ],
    )
    def test_response(
        self,
        answer_format,
        response_schema,
        raw_answer,
        parsed_answer,
        openai_client_mock,
        openai_client_config,
        chat_completion_mock,
        monkeypatch,
        mocker,
    ):
        mocker.patch("openai.types.chat.chat_completion", new_callable=AsyncMock)
        mocker.patch("openai.types.chat.parsed_chat_completion", new_callable=AsyncMock)

        openai_client = OpenAIProvider(config=openai_client_config)
        monkeypatch.setattr(openai_client, "_OpenAIProvider__openai_client", openai_client_mock)
        prompt = "hello world"
        prompts = [
            {"role": "system", "content": "hello world"},
            {"role": "user", "content": "hello openai"},
        ]
        result = openai_client(prompt, answer_format=answer_format, response_schema=response_schema)
        assert isinstance(result, LLMResponseType)
        assert result.raw_answer == raw_answer
        if answer_format == AnswerFormat.STRUCTURED:
            assert isinstance(result.parsed_answer, PydanticBaseModel)
            assert AnswerModel(**result.parsed_answer.model_dump()) == parsed_answer
        else:
            assert result.parsed_answer == parsed_answer

        result = openai_client(prompts, answer_format=answer_format, response_schema=response_schema)
        if answer_format == AnswerFormat.STRUCTURED:
            call_args = chat_completion_mock.parse.call_args_list
        else:
            call_args = chat_completion_mock.create.call_args_list
        assert call_args[0][1]["model"] == "fake model"
        if answer_format == AnswerFormat.JSON:
            assert call_args[0][1]["messages"][0]["content"].endswith(json.dumps(response_schema, indent=4))
        else:
            assert call_args[0][1]["messages"] == [{"role": "system", "content": prompt}]
        assert call_args[1][1]["messages"] == prompts
        assert call_args[1][1]["temperature"] == 0.2
        assert call_args[1][1]["max_tokens"] == 1000
        assert call_args[1][1]["seed"] == 42


class TestLLMProviderFactory:
    def test_get_llm_provider_passes_non_config_params_to_openai(self, async_openai_mock):
        """
        Test that arbitrary parameters are passed to AsyncOpenAI when creating
        an OpenAI provider through get_llm_provider.
        """
        # Call get_llm_provider with arbitrary parameters
        get_llm_provider(
            "openai",
            model="gpt-4",
            api_key="test_key",
            api_type="open_ai",
            default_headers={"X-Custom": "value"},
            timeout=30,
            max_retries=5,
            organization="test_org",
            custom_param="custom_value",
        )

        # Verify AsyncOpenAI was called with both standard and arbitrary parameters
        async_openai_mock.assert_called_once()
        call_args = async_openai_mock.call_args[1]

        # Check standard parameters
        assert call_args["api_key"] == "test_key"

        # Check arbitrary parameters
        assert call_args["default_headers"] == {"X-Custom": "value"}
        assert call_args["timeout"] == 30
        assert call_args["max_retries"] == 5
        assert call_args["organization"] == "test_org"
        assert "custom_param" not in call_args

    def test_get_llm_provider_passes_non_config_params_to_azure_openai(self, async_azure_openai_mock):
        """
        Test that arbitrary parameters are passed to AsyncAzureOpenAI when creating
        an Azure OpenAI provider through get_llm_provider.
        """
        # Call get_llm_provider with arbitrary parameters for Azure
        get_llm_provider(
            "openai",
            model="gpt-4",
            api_key="azure_key",
            api_type="azure",
            api_base="https://example.azure.openai.com",
            api_version="2023-05-15",
            default_headers={"X-Azure-Custom": "value"},
            timeout=60,
            max_retries=3,
            azure_custom_param="azure_value",
        )

        # Verify AsyncAzureOpenAI was called with both standard and arbitrary parameters
        async_azure_openai_mock.assert_called_once()
        call_args = async_azure_openai_mock.call_args[1]

        # Check standard parameters
        assert call_args["api_key"] == "azure_key"
        assert call_args["azure_endpoint"] == "https://example.azure.openai.com"
        assert call_args["api_version"] == "2023-05-15"

        # Check arbitrary parameters
        assert call_args["default_headers"] == {"X-Azure-Custom": "value"}
        assert call_args["timeout"] == 60
        assert call_args["max_retries"] == 3
        assert "azure_custom_param" not in call_args
