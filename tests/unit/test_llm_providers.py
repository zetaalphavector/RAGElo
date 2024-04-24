from unittest.mock import AsyncMock

import pytest

from ragelo import get_llm_provider
from ragelo.llm_providers.openai_client import OpenAIProvider


class TestOpenAIProvider:
    def test_response(
        self,
        chat_completion_mock,
        openai_client_mock,
        openai_client_config,
        monkeypatch,
        mocker,
    ):
        mocker.patch("openai.types.chat.chat_completion", new_callable=AsyncMock)

        openai_client = OpenAIProvider(config=openai_client_config)
        monkeypatch.setattr(
            openai_client, "_OpenAIProvider__openai_client", openai_client_mock
        )

        prompt = "hello world"
        prompts = [
            {"role": "system", "content": "hello world"},
            {"role": "user", "content": "hello openai"},
        ]
        result = openai_client(prompt)
        assert result == "fake response"
        result = openai_client(prompts)
        assert result == "fake response"

        call_args = chat_completion_mock.create.call_args_list
        assert call_args[0][1]["model"] == "fake model"
        assert call_args[0][1]["messages"] == [{"role": "system", "content": prompt}]
        assert call_args[1][1]["messages"] == prompts

    @pytest.mark.asyncio
    async def test_parallel_call(
        self,
        chat_completion_mock,
        openai_client_mock,
        openai_client_config,
        monkeypatch,
        mocker,
    ):
        mocker.patch("openai.types.chat.chat_completion", new_callable=AsyncMock)

        openai_client = OpenAIProvider(config=openai_client_config)
        monkeypatch.setattr(
            openai_client, "_OpenAIProvider__openai_client", openai_client_mock
        )

        prompt = "hello world"
        prompts = [
            {"role": "system", "content": "hello world"},
            {"role": "user", "content": "hello openai"},
        ]
        result = await openai_client.call_async(prompt)
        assert result == "fake response"
        result = await openai_client.call_async(prompts)
        assert result == "fake response"

        call_args = chat_completion_mock.create.call_args_list
        assert call_args[0][1]["model"] == "fake model"
        assert call_args[0][1]["messages"] == [{"role": "system", "content": prompt}]
        assert call_args[1][1]["messages"] == prompts

    def test_get_by_name(self, openai_client_config, openai_client_mock):
        provider = get_llm_provider("openai", api_key="fake key")
        assert isinstance(provider, OpenAIProvider)
