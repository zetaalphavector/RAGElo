import pytest


@pytest.fixture
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
    monkeypatch.setenv("OPENAI_GPT_DEPLOYMENT", "test_model")


@pytest.fixture
def mock_openai_api(monkeypatch):
    def mock_completion_create(*args, **kwargs):
        return {"choices": [{"message": {"content": "Mocked completion response"}}]}

    def mock_chat_completion_create(*args, **kwargs):
        return {"choices": [{"message": {"content": "Mocked chat response"}}]}

    monkeypatch.setattr("openai.Completion.create", mock_completion_create)
    monkeypatch.setattr("openai.ChatCompletion.create", mock_chat_completion_create)
