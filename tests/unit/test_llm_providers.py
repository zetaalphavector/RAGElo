from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel as PydanticBaseModel

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
