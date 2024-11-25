from __future__ import annotations

import json
from unittest.mock import AsyncMock

from pydantic import BaseModel as PydanticBaseModel

from ragelo import get_llm_provider
from ragelo.llm_providers.openai_client import OpenAIProvider
from ragelo.types.formats import AnswerFormat


class TestOpenAIProvider:
    def test_response_text(
        self,
        openai_client_mock_text,
        openai_client_config,
        chat_completion_mock_text,
        monkeypatch,
        mocker,
    ):
        mocker.patch("openai.types.chat.chat_completion", new_callable=AsyncMock)

        openai_client = OpenAIProvider(config=openai_client_config)
        monkeypatch.setattr(openai_client, "_OpenAIProvider__openai_client", openai_client_mock_text)

        prompt = "hello world"
        prompts = [
            {"role": "system", "content": "hello world"},
            {"role": "user", "content": "hello openai"},
        ]
        result = openai_client(prompt, answer_format=AnswerFormat.TEXT)
        assert result == "fake response"
        result = openai_client(prompts)
        assert result == "fake response"

        call_args = chat_completion_mock_text.create.call_args_list
        assert call_args[0][1]["model"] == "fake model"
        assert call_args[0][1]["messages"] == [{"role": "system", "content": prompt}]
        assert call_args[1][1]["messages"] == prompts

    def test_response_json(
        self,
        openai_client_mock_json,
        openai_client_config,
        chat_completion_mock_json,
        monkeypatch,
        mocker,
    ):
        mocker.patch("openai.types.chat.chat_completion", new_callable=AsyncMock)

        openai_client = OpenAIProvider(config=openai_client_config)
        monkeypatch.setattr(openai_client, "_OpenAIProvider__openai_client", openai_client_mock_json)

        prompt = "hello world"
        schema = {"KeyA": "Value of key a", "KeyB": "Value of key b"}
        result = openai_client(
            prompt,
            answer_format=AnswerFormat.JSON,
            response_schema=schema,
        )

        assert result == {"keyA": "valueA", "keyB": "valueB"}

        call_args = chat_completion_mock_json.create.call_args_list
        assert call_args[0][1]["model"] == "fake model"
        assert call_args[0][1]["messages"][0]["content"].endswith(json.dumps(schema, indent=4))

    def test_response_structured(
        self,
        openai_beta_client_mock,
        openai_client_config,
        monkeypatch,
        mocker,
    ):
        mocker.patch("openai.types.chat.parsed_chat_completion", new_callable=AsyncMock)

        openai_client = OpenAIProvider(config=openai_client_config)
        monkeypatch.setattr(openai_client, "_OpenAIProvider__openai_client", openai_beta_client_mock)

        class AnswerModel(PydanticBaseModel):
            keyA: str
            keyB: str

        prompt = "hello world"
        schema = AnswerModel
        result = openai_client(prompt, answer_format=AnswerFormat.STRUCTURED, response_schema=schema)
        try:
            result = AnswerModel(**result.model_dump())  # type: ignore
        except Exception as e:
            print(e)
            print(result)
            assert False
        assert isinstance(result, PydanticBaseModel)
        assert isinstance(result, AnswerModel)
        assert result.keyA == "valueA"
        assert result.keyB == "valueB"

        call_args = openai_beta_client_mock.beta.chat.completions.parse.call_args_list
        assert call_args[0][1]["model"] == "fake model"
        assert call_args[0][1]["messages"][0]["content"] == prompt

    def test_get_by_name(self, openai_client_config, openai_client_mock_text):
        provider = get_llm_provider("openai", api_key="fake key")
        assert isinstance(provider, OpenAIProvider)
