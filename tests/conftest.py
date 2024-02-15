import json
from unittest.mock import Mock

import pytest
from openai import OpenAI
from openai.resources.chat import Chat
from openai.resources.chat.completions import Completions

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.llm_providers.openai_client import OpenAIConfiguration
from ragelo.types.configurations import (
    AnswerEvaluatorConfig,
    LLMProviderConfiguration,
    RetrievalEvaluatorConfig,
)


@pytest.fixture
def openai_client_config():
    return OpenAIConfiguration(
        api_key="fake key",
        openai_org="fake org",
        openai_api_type="open_ai",
        openai_api_base=None,
        openai_api_version=None,
    )


@pytest.fixture
def llm_provider_config():
    return LLMProviderConfiguration(
        api_key="fake key",
    )


@pytest.fixture
def chat_completion_mock(mocker):
    return mocker.Mock(Completions)


@pytest.fixture
def openai_client_mock(mocker, chat_completion_mock):
    openai_client = mocker.Mock(OpenAI)
    type(openai_client).chat = mocker.Mock(Chat)
    type(openai_client.chat).completions = mocker.PropertyMock(
        return_value=chat_completion_mock
    )
    return openai_client


@pytest.fixture
def retrieval_eval_config():
    return RetrievalEvaluatorConfig(
        documents_path="tests/data/documents.csv",
        query_path="tests/data/queries.csv",
        output_file="tests/data/output.csv",
        force=True,
        verbose=True,
    )


@pytest.fixture
def answer_eval_config():
    return AnswerEvaluatorConfig(
        answers_file="tests/data/answers.csv",
        query_path="tests/data/queries.csv",
        output_file="tests/data/output_answers.csv",
        reasoning_file="tests/data/reasonings.csv",
        force=True,
        verbose=True,
    )


@pytest.fixture
def expert_retrieval_eval_config():
    return RetrievalEvaluatorConfig(
        documents_path="tests/data/documents.csv",
        query_path="tests/data/queries.csv",
        output_file="tests/data/output.csv",
        force=True,
        verbose=True,
        domain_long="Computer Science",
        domain_short="computer scientists",
        company="Zeta Alpha",
        extra_guidelines="-Super precise answers only!",
    )


@pytest.fixture
def rdnam_config():
    return RetrievalEvaluatorConfig(
        documents_path="tests/data/documents.csv",
        query_path="tests/data/queries.csv",
        output_file="tests/data/output.csv",
        force=True,
        role="You are a search quality rater evaluating the relevance of web pages. ",
        aspects=True,
        multiple=True,
        narrative_file="tests/data/rdnam_narratives.csv",
        description_file="tests/data/rdnam_descriptions.csv",
    )


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, config):
        self.config = config

    @classmethod
    def from_configuration(cls, config: LLMProviderConfiguration):
        return cls(config)

    def inner_call(self, prompt) -> str:
        return f"Processed {prompt}"

    def __call__(self, prompt) -> str:
        return self.inner_call(prompt)


@pytest.fixture
def llm_provider_mock(llm_provider_config):
    provider = MockLLMProvider(llm_provider_config)
    provider.inner_call = Mock(side_effect=lambda prompt: f"Processed {prompt}")
    return provider


@pytest.fixture
def llm_provider_answer_mock(llm_provider_config):
    provider = MockLLMProvider(llm_provider_config)
    provider.inner_call = Mock(
        side_effect=[
            "Agent [[A]] is better",
            "Agent [[B]] is better",
            "A tie. Therefore, [[C]]",
            "I don't know. [[C]]",
        ]
    )
    return provider


@pytest.fixture
def llm_provider_mock_mock(mocker):
    return mocker.Mock(MockLLMProvider)


@pytest.fixture
def llm_provider_domain_expert_mock(llm_provider_config):
    provider = MockLLMProvider(llm_provider_config)
    provider.inner_call = Mock(
        side_effect=["Reasoning answer", "2", "Reasoning_answer 2", "0"]
    )
    return provider


@pytest.fixture
def llm_provider_mock_rdnam(llm_provider_config):
    mocked_scores = [{"M": 2, "T": 1, "O": 1}, {"M": 1, "T": 1, "O": 2}]
    provider = MockLLMProvider(llm_provider_config)
    provider.inner_call = Mock(side_effect=lambda _: json.dumps(mocked_scores)[2:])
    return provider
