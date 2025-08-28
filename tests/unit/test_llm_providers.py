import pytest
from pydantic import BaseModel

from ragelo.llm_providers.openai_client import OpenAIProvider
from ragelo.types.formats import LLMInputPrompt, LLMResponseType


class AnswerModel(BaseModel):
    keyA: str
    keyB: str


class TestOpenAIProvider:
    @pytest.mark.parametrize(
        "response_schema,raw_answer,parsed_answer",
        [
            (None, "fake response", "fake response"),
            (
                {"keyA": "Value of key a", "KeyB": "Value of key b"},
                '{"keyA": "valueA", "keyB": "valueB"}',
                {"keyA": "valueA", "keyB": "valueB"},
            ),
            (
                AnswerModel,
                '{"keyA": "valueA", "keyB": "valueB"}',
                AnswerModel(keyA="valueA", keyB="valueB"),
            ),
        ],
    )
    def test_response(
        self,
        response_schema,
        raw_answer,
        parsed_answer,
        openai_client_mock,
        openai_client_config,
        monkeypatch,
    ):
        openai_client = OpenAIProvider(config=openai_client_config)
        monkeypatch.setattr(openai_client, "_OpenAIProvider__openai_client", openai_client_mock)

        # Test with just user prompt
        user_prompt = "hello world"
        result = openai_client(
            LLMInputPrompt(user_message=user_prompt),
            response_schema=response_schema,
        )
        assert isinstance(result, LLMResponseType)
        assert result.raw_answer == raw_answer
        is_structured = isinstance(response_schema, type) and issubclass(response_schema, BaseModel)
        if is_structured:
            assert isinstance(result.parsed_answer, BaseModel)
            assert AnswerModel(**result.parsed_answer.model_dump()) == parsed_answer
        else:
            assert result.parsed_answer == parsed_answer

        # Test with system and user prompts
        system_prompt = "hello system"
        user_prompt_2 = "hello openai"
        result = openai_client(
            LLMInputPrompt(user_message=user_prompt_2, system_prompt=system_prompt),
            response_schema=response_schema,
        )

        # Verify the correct API methods were called
        if is_structured:
            assert openai_client_mock.responses.parse.called
            call_args = openai_client_mock.responses.parse.call_args_list
            assert call_args[0][1]["model"] == "fake model"
            assert call_args[0][1]["input"] == user_prompt
            assert call_args[1][1]["input"] == user_prompt_2
            assert call_args[1][1]["instructions"] == system_prompt
        else:
            assert openai_client_mock.responses.create.called
            call_args = openai_client_mock.responses.create.call_args_list
            assert call_args[0][1]["model"] == "fake model"
            assert call_args[0][1]["input"] == user_prompt
            assert call_args[1][1]["input"] == user_prompt_2
            assert call_args[1][1]["instructions"] == system_prompt
