import json
from unittest.mock import Mock

import pytest
from openai import OpenAI
from openai.resources.chat import Chat
from openai.resources.chat.completions import Completions

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.llm_providers.openai_client import OpenAIConfiguration
from ragelo.types.configurations import (
    BaseEvaluatorConfig,
    CustomPromptAnswerEvaluatorConfig,
    CustomPromptEvaluatorConfig,
    DomainExpertEvaluatorConfig,
    FewShotEvaluatorConfig,
    FewShotExample,
    LLMProviderConfig,
    PairwiseEvaluatorConfig,
    RDNAMEvaluatorConfig,
)
from ragelo.utils import (
    load_answers_from_csv,
    load_queries_from_csv,
    load_retrieved_docs_from_csv,
)


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, config):
        self.config = config
        self.call_mocker = Mock(
            side_effect=lambda prompt: f"Received prompt: {prompt}. "
        )

    @classmethod
    def from_configuration(cls, config: LLMProviderConfig):
        return cls(config)

    # def inner_call(self, prompt) -> str:
    #     return f"Received prompt: {prompt}. "

    def __call__(self, prompt) -> str:
        return self.call_mocker(prompt)


@pytest.fixture
def queries_test():
    return load_queries_from_csv("tests/data/queries.csv")


@pytest.fixture
def qs_with_docs(queries_test):
    return load_retrieved_docs_from_csv(
        "tests/data/documents.csv", queries=queries_test
    )


@pytest.fixture
def answers_test(queries_test):
    return load_answers_from_csv("tests/data/answers.csv", queries=queries_test)


@pytest.fixture
def openai_client_config():
    return OpenAIConfiguration(
        api_key="fake key",
        org="fake org",
        api_type="open_ai",
        api_base=None,
        api_version=None,
        model_name="fake model",
    )


@pytest.fixture
def llm_provider_config():
    return LLMProviderConfig(
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
    return BaseEvaluatorConfig(
        documents_path="tests/data/documents.csv",
        query_path="tests/data/queries.csv",
        output_file="tests/data/output.csv",
        force=True,
        verbose=True,
    )


@pytest.fixture
def pairwise_answer_eval_config():
    return PairwiseEvaluatorConfig(
        output_file="tests/data/output_answers.csv",
        reasoning_path="tests/data/reasonings.csv",
        bidirectional=False,
        force=True,
        verbose=True,
    )


@pytest.fixture
def custom_answer_eval_config():
    return CustomPromptAnswerEvaluatorConfig(
        output_file="tests/data/output_answers.csv",
        prompt="""
You are an useful assistant for evaluating the quality of the answers generated \
by RAG Agents. Given the following retrieved documents and a user query, evaluate \
the quality of the answers based on their quality, trustworthiness and originality. \
The last line of your answer must be a json object with the keys "quality", \
"trustworthiness" and originality, each of them with a single number between 0 and \
2, where 2 is the highest score on that aspect.
DOCUMENTS RETRIEVED:
{docs_placeholder}

User Query: {query_placeholder}

Agent answer: {answer_placeholder}
""".strip(),
        query_placeholder="query_placeholder",
        answer_placeholder="answer_placeholder",
        reasoning_path="tests/data/reasonings.csv",
        documents_placeholder="docs_placeholder",
        force=True,
        verbose=True,
    )


@pytest.fixture
def expert_retrieval_eval_config():
    return DomainExpertEvaluatorConfig(
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
    return RDNAMEvaluatorConfig(
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


@pytest.fixture
def custom_prompt_retrieval_eval_config():
    return CustomPromptEvaluatorConfig(
        documents_path="tests/data/documents.csv",
        query_path="tests/data/queries.csv",
        output_file="tests/data/output.csv",
        force=True,
        verbose=True,
        prompt="query: {query_placeholder} doc: {document_placeholder}",
        query_placeholder="query_placeholder",
        document_placeholder="document_placeholder",
    )


@pytest.fixture
def few_shot_retrieval_eval_config():
    few_shot_samples = [
        FewShotExample(
            passage="Few shot example 1",
            query="Few shot query 1",
            relevance=2,
            reasoning="This is a good document",
        ),
        FewShotExample(
            passage="Few shot example 2",
            query="Few shot query 2",
            relevance=0,
            reasoning="This is a bad document",
        ),
    ]
    return FewShotEvaluatorConfig(
        documents_path="tests/data/documents.csv",
        query_path="tests/data/queries.csv",
        output_file="tests/data/output.csv",
        force=True,
        verbose=True,
        system_prompt="System prompt",
        few_shot_user_prompt="query: {query_placeholder} doc: {document_placeholder}",
        few_shot_assistant_answer=(
            '{reasoning_placeholder} {{"relevance": {relevance_placeholder}}}'
        ),
        query_placeholder="query_placeholder",
        document_placeholder="document_placeholder",
        reasoning_placeholder="reasoning_placeholder",
        relevance_placeholder="relevance_placeholder",
        few_shots=few_shot_samples,
    )


@pytest.fixture
def llm_provider_mock(llm_provider_config):
    return MockLLMProvider(llm_provider_config)


@pytest.fixture
def llm_provider_json_mock(llm_provider_config):
    provider = MockLLMProvider(llm_provider_config)
    provider.call_mocker = Mock(
        side_effect=lambda _: f"LLM JSON response" f'\n{{"relevance": 0}}'
    )
    return provider


@pytest.fixture
def llm_provider_pairwise_answer_mock(llm_provider_config):
    provider = MockLLMProvider(llm_provider_config)
    provider.call_mocker = Mock(
        side_effect=[
            "Agent [[A]] is better",
            "Agent [[B]] is better",
            "A tie. Therefore, [[C]]",
            "I don't know. [[C]]",
        ]
    )
    return provider


@pytest.fixture
def llm_provider_answer_mock(llm_provider_config):
    provider = MockLLMProvider(llm_provider_config)
    provider.inner_call = Mock(
        side_effect=lambda prompt: f"Answer for {prompt}\n"
        '{"quality": 2, "trustworthiness": 1, "originality": 1}',
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
