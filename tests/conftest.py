from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI
from openai.resources.beta import AsyncBeta
from openai.resources.beta.chat import AsyncChat
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.parsed_chat_completion import (
    ParsedChatCompletion,
    ParsedChatCompletionMessage,
    ParsedChoice,
)
from pydantic import BaseModel as PydanticBaseModel

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.llm_providers.openai_client import OpenAIConfiguration
from ragelo.types.configurations import (
    BaseEvaluatorConfig,
    BaseRetrievalEvaluatorConfig,
    CustomPromptAnswerEvaluatorConfig,
    CustomPromptEvaluatorConfig,
    DomainExpertEvaluatorConfig,
    FewShotEvaluatorConfig,
    LLMProviderConfig,
    PairwiseEvaluatorConfig,
    RDNAMEvaluatorConfig,
)
from ragelo.types.configurations.retrieval_evaluator_configs import FewShotExample
from ragelo.types.formats import AnswerFormat, LLMResponseType
from ragelo.types.results import (
    AnswerEvaluatorResult,
    EloTournamentResult,
    RetrievalEvaluatorResult,
)


@pytest.fixture
def openai_client_config():
    return OpenAIConfiguration(
        api_key="fake key",
        org="fake org",
        api_type="open_ai",
        api_base=None,
        api_version=None,
        model="fake model",
    )


@pytest.fixture
def llm_provider_config():
    return LLMProviderConfig(
        api_key="fake key",
    )


class AnswerModel(PydanticBaseModel):
    keyA: str
    keyB: str


def answer_model_factory(prompt, answer_format, **kwargs):
    if answer_format == AnswerFormat.STRUCTURED:
        return LLMResponseType(
            raw_answer='{"keyA": "valueA", "keyB": "valueB"}',
            parsed_answer=AnswerModel(keyA="valueA", keyB="valueB"),
        )
    elif answer_format == AnswerFormat.JSON:
        return LLMResponseType(
            raw_answer='{"relevance": 1}',
            parsed_answer={"relevance": 1},
        )
    elif answer_format == AnswerFormat.TEXT:
        return LLMResponseType(
            raw_answer='{"keyA": "valueA", "keyB": "valueB"}',
            parsed_answer='{"keyA": "valueA", "keyB": "valueB"}',
        )
    else:
        raise ValueError(f"Unsupported answer format: {answer_format}")


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, config):
        self.config = config
        self.async_call_mocker = AsyncMock(side_effect=answer_model_factory)

    @classmethod
    def from_configuration(cls, config: LLMProviderConfig):
        return cls(config)

    async def call_async(self, prompt, answer_format, *args, **kwargs):
        return await self.async_call_mocker(prompt, answer_format)


@pytest.fixture
def chat_completion_mock(answer_format):
    if answer_format == AnswerFormat.STRUCTURED:
        _cls = ParsedChatCompletion
    else:
        _cls = ChatCompletion

    async_mock = AsyncMock()
    fake_response = _cls(
        id="fake id",
        choices=[],
        created=0,
        model="fake model",
        object="chat.completion",
    )
    if answer_format == AnswerFormat.TEXT:
        fake_response.choices.append(
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(content="fake response", role="assistant"),
            )  # type: ignore
        )
        async_mock.create.return_value = fake_response
    elif answer_format == AnswerFormat.JSON:
        fake_response.choices.append(
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(content='{"keyA": "valueA", "keyB": "valueB"}', role="assistant"),
            )  # type: ignore
        )
        async_mock.create.return_value = fake_response
    elif answer_format == AnswerFormat.STRUCTURED:
        fake_response.choices.append(
            ParsedChoice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ParsedChatCompletionMessage(
                    content='{"keyA": "valueA", "keyB": "valueB"}',
                    parsed=AnswerModel(keyA="valueA", keyB="valueB"),
                    role="assistant",
                ),
            )  # type: ignore
        )
        async_mock.parse.return_value = fake_response
    else:
        raise ValueError(f"Unsupported answer format: {answer_format}")

    return async_mock


@pytest.fixture
def openai_client_mock(mocker, chat_completion_mock):
    openai_client = mocker.AsyncMock(AsyncOpenAI)
    type(openai_client).chat = mocker.AsyncMock(AsyncChat)
    type(openai_client.chat).completions = mocker.PropertyMock(return_value=chat_completion_mock)
    type(openai_client).beta = mocker.AsyncMock(AsyncBeta)
    type(openai_client.beta).chat = mocker.AsyncMock(AsyncChat)
    type(openai_client.beta.chat).completions = mocker.PropertyMock(return_value=chat_completion_mock)
    return openai_client


@pytest.fixture
def base_experiment_config():
    config = {
        "experiment_name": "test_experiment",
        "persist_on_disk": False,
        "queries_csv_path": "tests/data/queries.csv",
        "documents_csv_path": "tests/data/documents.csv",
        "answers_csv_path": "tests/data/answers.csv",
        "rich_print": True,
        "verbose": True,
    }
    return config


@pytest.fixture
def experiment(base_experiment_config):
    from ragelo.types.experiment import Experiment

    return Experiment(**base_experiment_config)


@pytest.fixture
def experiment_with_retrieval_scores(experiment):
    experiment.queries["0"].retrieved_docs["0"].retrieved_by["agent1"] = 1.0
    experiment.queries["0"].retrieved_docs["0"].retrieved_by["agent2"] = 0.5
    experiment.queries["0"].retrieved_docs["1"].retrieved_by["agent1"] = 0.5
    experiment.queries["0"].retrieved_docs["1"].retrieved_by["agent2"] = 1.0
    experiment.queries["1"].retrieved_docs["2"].retrieved_by["agent1"] = 1.0
    experiment.queries["1"].retrieved_docs["2"].retrieved_by["agent2"] = 0.5
    experiment.queries["1"].retrieved_docs["3"].retrieved_by["agent1"] = 0.5
    experiment.queries["1"].retrieved_docs["3"].retrieved_by["agent2"] = 1.0
    return experiment


@pytest.fixture
def empty_experiment():
    from ragelo.types.experiment import Experiment

    return Experiment(experiment_name="test_experiment")


@pytest.fixture
def retrieval_evaluation():
    return RetrievalEvaluatorResult(
        qid="0",
        did="0",
        raw_answer="Document is relevant. Score: 1.0",
        answer=1,
    )


@pytest.fixture
def answer_evaluation():
    return AnswerEvaluatorResult(
        qid="0",
        agent="agent1",
        pairwise=False,
        raw_answer="Answer is good. Scores: {'quality': 1.0, 'relevance': 0.8}",
        answer={"quality": 1.0, "relevance": 0.8},
    )


@pytest.fixture
def pairwise_answer_evaluation():
    return AnswerEvaluatorResult(
        qid="0",
        agent_a="agent1",
        agent_b="agent2",
        pairwise=True,
        raw_answer="Answer [[A]] is better than [[B]]",
        answer="A",
    )


@pytest.fixture
def elo_tournament_result():
    return EloTournamentResult(
        agents=["agent1", "agent2"],
        scores={"agent1": 1200, "agent2": 1000},
        games_played={"agent1": 1, "agent2": 1},
        wins={"agent1": 1, "agent2": 0},
        loses={"agent1": 0, "agent2": 1},
        ties={"agent1": 0, "agent2": 0},
        total_games=2,
        total_tournaments=1,
        std_dev={"agent1": 0, "agent2": 0},
    )


@pytest.fixture
def base_eval_config():
    return BaseEvaluatorConfig(force=True, verbose=True, llm_answer_format=AnswerFormat.JSON)


@pytest.fixture
def base_retrieval_eval_config(base_eval_config):
    base_config = base_eval_config.model_dump()
    return BaseRetrievalEvaluatorConfig(
        **base_config,
    )


@pytest.fixture
def pairwise_answer_eval_config(base_eval_config):
    base_config = base_eval_config.model_dump()
    base_config["document_evaluations_file"] = "tests/data/reasonings.csv"
    config = PairwiseEvaluatorConfig(
        bidirectional=True,
        **base_config,
    )
    return config


@pytest.fixture
def custom_answer_eval_config(base_eval_config):
    base_config = base_eval_config.model_dump()
    base_config["answer_format_answer_evaluator"] = "multi_field_json"
    config = CustomPromptAnswerEvaluatorConfig(
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
        **base_config,
    )
    return config


@pytest.fixture
def expert_retrieval_eval_config(base_eval_config):
    base_eval_config = base_eval_config.model_dump()

    return DomainExpertEvaluatorConfig(
        expert_in="Computer Science",
        domain_short="computer scientists",
        company="Zeta Alpha",
        extra_guidelines=["Super precise answers only!"],
        **base_eval_config,
    )


@pytest.fixture
def rdnam_config(base_eval_config):
    base_config = base_eval_config.model_dump()
    base_config["query_file"] = "tests/data/rdnam_queries.csv"
    return RDNAMEvaluatorConfig(
        annotator_role="You are a search quality rater evaluating the relevance of web pages. ",
        use_multiple_annotators=True,
        use_aspects=True,
        **base_config,
    )


@pytest.fixture
def custom_prompt_retrieval_eval_config(base_eval_config):
    base_eval_config = base_eval_config.model_dump()
    config = CustomPromptEvaluatorConfig(
        prompt="query: {query} doc: {document}",
        **base_eval_config,
    )
    return config


@pytest.fixture
def llm_provider_mock(llm_provider_config):
    return MockLLMProvider(llm_provider_config)


@pytest.fixture
def llm_provider_mock_rdnam(llm_provider_config):
    mocked_scores = {
        "annotator_1": {"intent_match": 2, "trustworthiness": 1, "overall": 1},
        "annotator_2": {"intent_match": 1, "trustworthiness": 1, "overall": 2},
    }
    LLM_response = LLMResponseType(raw_answer=json.dumps(mocked_scores)[2:], parsed_answer=mocked_scores)
    provider = MockLLMProvider(llm_provider_config)

    def side_effect(*args, **kwargs):
        return LLM_response

    provider.async_call_mocker = AsyncMock(side_effect=side_effect)
    return provider


@pytest.fixture
def llm_provider_reasoner_mock(llm_provider_config):
    provider = MockLLMProvider(llm_provider_config)
    answer = LLMResponseType(raw_answer="The document is very relevant", parsed_answer="The document is very relevant")

    def side_effect(*args, **kwargs):
        return answer

    provider.async_call_mocker = AsyncMock(side_effect=side_effect)
    return provider


@pytest.fixture
def llm_provider_pairwise_answer_mock(llm_provider_config):
    provider = MockLLMProvider(llm_provider_config)
    provider.async_call_mocker = AsyncMock(
        side_effect=[
            "Async Agent [[A]] is better",
            "Async Agent [[B]] is better",
            "Async A tie. Therefore, [[C]]",
            "Async I don't know. [[C]]",
        ]
    )
    return provider


@pytest.fixture
def llm_provider_answer_mock(llm_provider_config):
    provider = MockLLMProvider(llm_provider_config)
    provider.async_call_mocker = AsyncMock(
        side_effect=lambda prompt: f"Async answer for {prompt}\n"
        '{"quality": 1, "trustworthiness": 0, "originality": 0}',
    )
    return provider


# from ragelo.types.configurations.retrieval_evaluator_configs import FewShotExample

# from ragelo.utils import (
# add_answers_from_csv,
# add_documents_from_csv,
# load_queries_from_csv,
# )


# @pytest.fixture
# def queries_test():
#     return load_queries_from_csv("tests/data/queries.csv")


# @pytest.fixture
# def rdnam_queries():
#     queries = load_queries_from_csv("tests/data/rdnam_queries.csv")
#     return add_documents_from_csv("tests/data/documents.csv", queries=queries)


# @pytest.fixture
# def answers_test(queries_test):
#     return add_answers_from_csv("tests/data/answers.csv", queries=queries_test)


@pytest.fixture
def few_shot_retrieval_eval_config(base_eval_config):
    base_eval_config = base_eval_config.model_dump()
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
        **base_eval_config,
    )
