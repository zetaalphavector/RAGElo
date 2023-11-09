import pytest

from ragelo.utils.openai_client import OpenAiClient


def test_openai_client_call_with_string_prompt(mock_env_vars, mock_openai_api):
    client = OpenAiClient()
    response = client("Test prompt")
    assert response == "Mocked chat response"


def test_openai_client_call_with_list_prompt(mock_env_vars, mock_openai_api):
    client = OpenAiClient()
    response = client([{"role": "system", "content": "Test prompt"}])
    assert response == "Mocked chat response"
