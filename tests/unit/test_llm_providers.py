from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
)

from ragelo import get_llm_provider
from ragelo.llm_providers.openai_client import OpenAIProvider


class TestOpenAIProvider:
    def test_response(
        self, chat_completion_mock, openai_client_mock, openai_client_config
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
        openai_client.set_openai_client(openai_client_mock)

        prompt = "hello world"
        prompts = [
            {"role": "system", "content": "hello world"},
            {"role": "user", "content": "hello openai"},
        ]
        result_1 = openai_client(prompt)
        openai_client(prompts)
        call_args = chat_completion_mock.create.call_args_list
        assert call_args[0][1]["model"] == "fake model"
        assert call_args[0][1]["messages"] == [{"role": "system", "content": prompt}]
        assert call_args[1][1]["messages"] == prompts
        assert result_1 == "fake response"

    def test_get_by_name(self, openai_client_config, openai_client_mock):
        provider = get_llm_provider("openai", api_key="fake key")
        assert isinstance(provider, OpenAIProvider)
