import asyncio

import aiohttp
from aioresponses import aioresponses
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from yarl import URL

from ragelo import get_llm_provider
from ragelo.llm_providers.openai_client import OpenAIProvider


class TestOpenAIProvider:
    def test_response(
        self,
        chat_completion_mock,
        openai_client_mock,
        openai_client_config,
        monkeypatch,
    ):
        chat_completion_mock.create.return_value = ChatCompletion(
            id="fake id",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    logprobs=None,
                    message=ChatCompletionMessage(
                        content="fake response", role="assistant"
                    ),
                )
            ],
            created=0,
            model="fake model",
            object="chat.completion",
        )
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

    def test_parallel_call(
        self,
        openai_client_config,
    ):
        openai_client = OpenAIProvider(config=openai_client_config)
        prompt = "hello world"
        prompts = [
            {"role": "system", "content": "hello world"},
            {"role": "user", "content": "hello openai"},
        ]

        with aioresponses() as m:
            with asyncio.Runner() as runner:
                session = aiohttp.ClientSession()
                m.post(
                    "https://api.openai.com/v1/chat/completions",
                    status=200,
                    payload={
                        "id": "requestID",
                        "object": "chat.completion",
                        "created": 1713271118,
                        "model": "gpt-4-turbo-2024-04-09",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "index": 0,
                                "logprobs": None,
                                "message": {
                                    "content": "fake response",
                                    "role": "assistant",
                                },
                            }
                        ],
                    },
                )
                result = runner.run(
                    openai_client.call_async(prompt=prompt, session=session)
                )

        assert result == "fake response"
        messages_sent = m.requests[
            ("POST", URL("https://api.openai.com/v1/chat/completions"))
        ][0].kwargs["json"]["messages"]
        assert len(messages_sent) == 1
        assert messages_sent[0]["content"] == prompt

    def test_get_by_name(self, openai_client_config, openai_client_mock):
        provider = get_llm_provider("openai", api_key="fake key")
        assert isinstance(provider, OpenAIProvider)
