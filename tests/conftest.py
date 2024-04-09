import json
from dataclasses import asdict
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
def rdnam_queries():
    queries = load_queries_from_csv("tests/data/rdnam_queries.csv")
    return load_retrieved_docs_from_csv("tests/data/documents.csv", queries=queries)


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
def base_eval_config():
    return BaseEvaluatorConfig(
        documents_path="tests/data/documents.csv",
        query_path="tests/data/queries.csv",
        output_file="tests/data/output.csv",
        force=True,
        verbose=True,
        write_output=False,
    )


@pytest.fixture
def pairwise_answer_eval_config(base_eval_config):
    config = PairwiseEvaluatorConfig(
        reasoning_path="tests/data/reasonings.csv",
        bidirectional=False,
        **asdict(base_eval_config),
    )
    return config


@pytest.fixture
def custom_answer_eval_config(base_eval_config):
    base_config = asdict(base_eval_config)
    del base_config["answer_format"]
    del base_config["scoring_key"]
    config = CustomPromptAnswerEvaluatorConfig(
        reasoning_path="tests/data/reasonings.csv",
        prompt="""
You are an useful assistant for evaluating the quality of the answers generated \
by RAG Agents. Given the following retrieved documents and a user query, evaluate \
the quality of the answers based on their quality, trustworthiness and originality. \
The last line of your answer must be a json object with the keys "quality", \
"trustworthiness" and originality, each of them with a single number between 0 and \
2, where 2 is the highest score on that aspect.
DOCUMENTS RETRIEVED:
{documents}

User Query: {query}

Agent answer: {answer}
""".strip(),
        answer_format="multi_field_json",
        **base_config,
    )
    return config


@pytest.fixture
def expert_retrieval_eval_config(base_eval_config):
    base_eval_config = asdict(base_eval_config)
    del base_eval_config["scoring_key"]
    del base_eval_config["answer_format"]
    return DomainExpertEvaluatorConfig(
        domain_long="Computer Science",
        domain_short="computer scientists",
        company="Zeta Alpha",
        extra_guidelines=["Super precise answers only!"],
        **base_eval_config,
    )


@pytest.fixture
def rdnam_config(base_eval_config):
    base_config = asdict(base_eval_config)
    base_config["query_path"] = "tests/data/rdnam_queries.csv"
    return RDNAMEvaluatorConfig(
        annotator_role="You are a search quality rater evaluating the relevance of web pages. ",
        use_multiple_annotators=True,
        **base_config,
    )


@pytest.fixture
def custom_prompt_retrieval_eval_config(base_eval_config):
    base_eval_config = asdict(base_eval_config)
    del base_eval_config["scoring_key"]
    config = CustomPromptEvaluatorConfig(
        prompt="query: {query} doc: {document}",
        scoring_key="relevance",
        **base_eval_config,
    )
    return config


@pytest.fixture
def few_shot_retrieval_eval_config(base_eval_config):
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
        system_prompt="System prompt",
        few_shot_user_prompt="query: {query} doc: {document}",
        few_shot_assistant_answer=('{reasoning} {{"relevance": {relevance}}}'),
        reasoning_placeholder="reasoning",
        relevance_placeholder="relevance",
        few_shots=few_shot_samples,
        **asdict(base_eval_config),
    )


@pytest.fixture
def llm_provider_mock(llm_provider_config):
    return MockLLMProvider(llm_provider_config)


@pytest.fixture
def llm_provider_json_mock(llm_provider_config):
    provider = MockLLMProvider(llm_provider_config)
    provider.call_mocker = Mock(
        side_effect=lambda _: 'LLM JSON response\n{"relevance": 0}'
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
    provider.call_mocker = Mock(
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
    provider.call_mocker = Mock(
        side_effect=["Reasoning answer", "2", "Reasoning_answer 2", "0"]
    )
    return provider


@pytest.fixture
def llm_provider_mock_rdnam(llm_provider_config):
    mocked_scores = [{"M": 2, "T": 1, "O": 1}, {"M": 1, "T": 1, "O": 2}]
    provider = MockLLMProvider(llm_provider_config)
    provider.call_mocker = Mock(side_effect=lambda _: json.dumps(mocked_scores)[2:])
    return provider
