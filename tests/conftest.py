from __future__ import annotations

from typing import Any, Type
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel

from ragelo.llm_providers.base_llm_provider import BaseLLMProvider
from ragelo.llm_providers.openai_client import OpenAIConfiguration
from ragelo.types.answer_formats import (
    PairWiseAnswerAnswerFormat,
    RDNAMAnswerEvaluatorFormat,
    RDNAMMultipleAnnotatorsAnswer,
    RetrievalAnswerEvaluatorFormat,
)
from ragelo.types.configurations import (
    BaseAnswerEvaluatorConfig,
    BaseEvaluatorConfig,
    BaseRetrievalEvaluatorConfig,
    CustomPromptAnswerEvaluatorConfig,
    CustomPromptEvaluatorConfig,
    DomainExpertEvaluatorConfig,
    FewShotEvaluatorConfig,
    LLMProviderConfig,
    PairwiseDomainExpertEvaluatorConfig,
    PairwiseEvaluatorConfig,
    RDNAMEvaluatorConfig,
)
from ragelo.types.configurations.retrieval_evaluator_configs import FewShotExample
from ragelo.types.evaluables import ChatMessage
from ragelo.types.experiment import Experiment
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.results import AnswerEvaluatorResult, EloTournamentResult, RetrievalEvaluatorResult
from ragelo.types.types import AnswerEvaluatorTypes, RetrievalEvaluatorTypes
from ragelo.utils import string_to_template


def pytest_addoption(parser):
    parser.addoption(
        "--runopenai",
        action="store_true",
        default=False,
        help="run tests that requires an OpenAI API key",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runopenai"):
        # --runopenai given in cli: do not skip tests that will call OpenAI
        return
    skip_openai = pytest.mark.skip(reason="need --runopenai option to run")
    for item in items:
        if "requires_openai" in item.keywords:
            item.add_marker(skip_openai)


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


def answer_model_factory(input: LLMInputPrompt, response_schema, **kwargs):
    # Mirror OpenAIProvider behavior: response format is inferred from response_schema
    if response_schema == RetrievalAnswerEvaluatorFormat:
        return LLMResponseType(
            raw_answer='{"reasoning": "The document is very relevant", "score": 2}',
            parsed_answer=response_schema(
                reasoning="The document is very relevant",
                score=2,
            ),
        )
    if response_schema == PairWiseAnswerAnswerFormat:
        return LLMResponseType(
            raw_answer='{"answer_a_analysis": "The document is very relevant", "answer_b_analysis": "The document is not relevant", "comparison_reasoning": "The document is more relevant for the user question", "winner": "A"}',
            parsed_answer=response_schema(
                answer_a_analysis="Answer A is good",
                answer_b_analysis="Answer B is bad",
                comparison_reasoning="Answer A is better than Answer B",
                winner="A",
            ),
        )
    if response_schema == AnswerFormat:
        return LLMResponseType(
            raw_answer='{"keyA": "valueA", "keyB": "valueB"}',
            parsed_answer=response_schema(keyA="valueA", keyB="valueB"),
        )
    if response_schema == AnswerEvaluatorFormat:
        return LLMResponseType(
            raw_answer='{"quality": 1, "trustworthiness": 0, "originality": 0}',
            parsed_answer=response_schema(quality=1, trustworthiness=0, originality=0),
        )
    elif isinstance(response_schema, dict):
        return LLMResponseType(
            raw_answer='{"score": 1.0}',
            parsed_answer={"score": 1.0},
        )
    else:
        return LLMResponseType(
            raw_answer='{"keyA": "valueA", "keyB": "valueB"}',
            parsed_answer='{"keyA": "valueA", "keyB": "valueB"}',
        )


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, config):
        super().__init__(config)
        self.async_call_mocker = AsyncMock()

    @classmethod
    def from_config(cls, config: LLMProviderConfig):
        return cls(config)

    async def call_async(
        self,
        input: LLMInputPrompt,
        response_schema: Type[BaseModel] | dict[str, Any] | None = None,
    ) -> LLMResponseType:
        # Record the call for assertions in tests
        if self.async_call_mocker.side_effect is not None:
            # If a custom side effect is set by a test/fixture, use it
            return await self.async_call_mocker(input, response_schema)
        else:
            # Still record the call (without relying on its return value)
            try:
                await self.async_call_mocker(input, response_schema)
            except Exception:
                pass
            return answer_model_factory(input, response_schema)


@pytest.fixture
def responses_api_mock(mocker, answer_format):
    """Factory helpers to mock the OpenAI Responses API.

    The OpenAIProvider chooses between responses.create (TEXT/JSON) and
    responses.parse (STRUCTURED) based on the provided response_schema.
    This fixture exposes helpers so the client mock can return suitable
    objects depending on call kwargs.
    """
    holder = mocker.Mock()

    def create_text_response():
        resp = mocker.Mock()
        resp.output_text = "fake response"
        return resp

    def create_json_response():
        resp = mocker.Mock()
        resp.output_text = '{"keyA": "valueA", "keyB": "valueB"}'
        return resp

    def parse_structured_response():
        resp = mocker.Mock()
        resp.output_text = '{"keyA": "valueA", "keyB": "valueB"}'
        resp.output_parsed = answer_format(keyA="valueA", keyB="valueB")
        return resp

    holder.create_text_response = create_text_response
    holder.create_json_response = create_json_response
    holder.parse_structured_response = parse_structured_response

    return holder


@pytest.fixture
def openai_client_mock(mocker, responses_api_mock):
    openai_client = mocker.AsyncMock(AsyncOpenAI)

    # Mock the responses API
    type(openai_client).responses = mocker.AsyncMock()

    # responses.create returns TEXT by default; if JSON schema is requested, return JSON
    def create_side_effect(*args, **kwargs):
        text = kwargs.get("text")
        text_fmt = kwargs.get("text_format")
        if isinstance(text, dict):
            return responses_api_mock.create_json_response()
        elif isinstance(text_fmt, type(BaseModel)):
            return responses_api_mock.parse_structured_response()
        return responses_api_mock.create_text_response()

    openai_client.responses.create = mocker.AsyncMock(side_effect=create_side_effect)
    openai_client.responses.parse = mocker.AsyncMock(side_effect=create_side_effect)

    return openai_client


@pytest.fixture
def base_experiment_config():
    config = {
        "experiment_name": "test_experiment",
        "save_on_disk": False,
        "cache_evaluations": False,
        "queries_csv_path": "tests/data/queries.csv",
        "documents_csv_path": "tests/data/documents.csv",
        "answers_csv_path": "tests/data/answers.csv",
        "rich_print": True,
        "verbose": True,
    }
    return config


@pytest.fixture
def experiment(base_experiment_config):
    return Experiment(**base_experiment_config)


@pytest.fixture
def experiment_with_conversations_and_reasonings(experiment):
    experiment.queries["0"].answers["agent1"].conversation = [
        ChatMessage(sender="user", content="What is the capital of Brazil?"),
        ChatMessage(
            sender="agent1",
            content="Brasília is the capital of Brazil, according to [0].",
        ),
    ]
    experiment.queries["0"].answers["agent2"].conversation = [
        ChatMessage(sender="user", content="What is the capital of Brazil?"),
        ChatMessage(
            sender="agent2",
            content="According to [1], Rio de Janeiro used to be the capital of Brazil, until the 60s.",
        ),
    ]
    experiment.queries["1"].answers["agent1"].conversation = [
        ChatMessage(sender="user", content="What is the capital of France?"),
        ChatMessage(sender="agent1", content="Paris is the capital of France, according to [2]."),
    ]
    experiment.queries["1"].answers["agent2"].conversation = [
        ChatMessage(sender="user", content="What is the capital of France?"),
        ChatMessage(
            sender="agent2",
            content="According to [3], Lyon is the second largest city in France. Meanwhile, Paris is its capital [2].",
        ),
    ]
    experiment.queries["0"].retrieved_docs["0"].evaluation = RetrievalEvaluatorResult(
        qid=experiment.queries["0"].qid,
        did="0",
        raw_answer="The document is very relevant as it directly answers the user's question about the capital of Brazil",
        answer="The document is very relevant as it directly answers the user's question about the capital of Brazil",
    )
    experiment.queries["0"].retrieved_docs["1"].evaluation = RetrievalEvaluatorResult(
        qid=experiment.queries["0"].qid,
        did="1",
        raw_answer="The document is somewhat relevant as it provides historical information about the capital of Brazil, but it does not provide the current capital.",
        answer="The document is somewhat relevant as it provides historical information about the capital of Brazil, but it does not provide the current capital.",
    )
    experiment.queries["1"].retrieved_docs["2"].evaluation = RetrievalEvaluatorResult(
        qid=experiment.queries["1"].qid,
        did="2",
        raw_answer="The document is very relevant as it directly answers the user's question about the capital of France.",
        answer="The document is very relevant as it directly answers the user's question about the capital of France.",
    )
    experiment.queries["1"].retrieved_docs["3"].evaluation = RetrievalEvaluatorResult(
        qid=experiment.queries["1"].qid,
        did="3",
        raw_answer="The document is not relevant to the user question as it does not provide information about the capital of France.",
        answer="The document is not relevant to the user question as it does not provide information about the capital of France.",
    )
    return experiment


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
def empty_experiment(base_experiment_config):
    base_config = base_experiment_config.copy()
    base_config.pop("queries_csv_path")
    base_config.pop("documents_csv_path")
    base_config.pop("answers_csv_path")
    return Experiment(**base_config)


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
    return BaseEvaluatorConfig(force=True, verbose=True)


@pytest.fixture
def base_retrieval_eval_config(base_eval_config):
    base_config = base_eval_config.model_dump(exclude_unset=True)
    base_config["evaluator_name"] = RetrievalEvaluatorTypes.REASONER
    return BaseRetrievalEvaluatorConfig(
        **base_config,
    )


@pytest.fixture
def custom_answer_eval_config(base_eval_config, answer_eval_format):
    base_config = base_eval_config.model_dump(exclude_unset=True)
    base_config["evaluator_name"] = AnswerEvaluatorTypes.CUSTOM_PROMPT
    config = CustomPromptAnswerEvaluatorConfig(
        llm_response_schema=answer_eval_format,
        user_prompt=string_to_template(
            """ 
You are an useful assistant for evaluating the quality of the answers generated \
by RAG Agents. Given the following retrieved documents and a user query, evaluate \
the quality of the answers based on their quality, trustworthiness and originality. \
The last line of your answer must be a json object with the keys "quality", \
"trustworthiness" and originality, each of them with a single number between 0 and \
2, where 2 is the highest score on that aspect.
DOCUMENTS RETRIEVED:
{% for document in documents %}
{{document.text}}
{% endfor %}
User Query: {{ query.query }}

Agent answer: {{answer.text}}"""
        ),
        **base_config,
    )
    return config


@pytest.fixture
def expert_retrieval_eval_config(base_eval_config):
    base_eval_config = base_eval_config.model_dump(exclude_unset=True)
    base_eval_config["evaluator_name"] = AnswerEvaluatorTypes.DOMAIN_EXPERT
    return DomainExpertEvaluatorConfig(
        expert_in="Computer Science",
        domain_short="computer scientists",
        company="Zeta Alpha",
        extra_guidelines=["Super precise answers only!"],
        **base_eval_config,
    )


@pytest.fixture
def rdnam_config(base_eval_config):
    base_config = base_eval_config.model_dump(exclude_unset=True)
    base_config["query_file"] = "tests/data/rdnam_queries.csv"
    base_config["evaluator_name"] = RetrievalEvaluatorTypes.RDNAM
    return RDNAMEvaluatorConfig(
        annotator_role="You are a search quality rater evaluating the relevance of web pages. ",
        use_multiple_annotators=True,
        use_aspects=True,
        **base_config,
    )


@pytest.fixture
def custom_prompt_retrieval_eval_config(base_eval_config):
    base_eval_config = base_eval_config.model_dump(exclude_unset=True)
    base_eval_config["evaluator_name"] = RetrievalEvaluatorTypes.CUSTOM_PROMPT
    config = CustomPromptEvaluatorConfig(
        system_prompt="CUSTOM PROMPT SYSTEM PROMPT",
        user_prompt="query: {{ query.query }} doc: {{ document.text }}",
        **base_eval_config,
    )
    return config


@pytest.fixture
def llm_provider_mock(llm_provider_config):
    return MockLLMProvider(llm_provider_config)


@pytest.fixture
def llm_provider_mock_retrieval(llm_provider_config):
    mocked_answer = RetrievalAnswerEvaluatorFormat(
        reasoning="The document is very relevant",
        score=2,
    )
    LLM_response = LLMResponseType(
        raw_answer=mocked_answer.model_dump_json(),
        parsed_answer=mocked_answer,
    )
    provider = MockLLMProvider(llm_provider_config)
    provider.async_call_mocker = AsyncMock(side_effect=lambda *args, **kwargs: LLM_response)
    return provider


@pytest.fixture
def llm_provider_mock_rdnam(llm_provider_config):
    mocked_answer = RDNAMMultipleAnnotatorsAnswer(
        annotator_1=RDNAMAnswerEvaluatorFormat(intent_match=2, trustworthiness=1, overall=1),
        annotator_2=RDNAMAnswerEvaluatorFormat(intent_match=1, trustworthiness=1, overall=2),
        annotator_3=RDNAMAnswerEvaluatorFormat(intent_match=1, trustworthiness=1, overall=1),
        annotator_4=RDNAMAnswerEvaluatorFormat(intent_match=0, trustworthiness=0, overall=0),
        annotator_5=RDNAMAnswerEvaluatorFormat(intent_match=1, trustworthiness=1, overall=2),
    )
    LLM_response = LLMResponseType(raw_answer=mocked_answer.model_dump_json(), parsed_answer=mocked_answer)

    LLM_response = LLMResponseType(raw_answer=mocked_answer.model_dump_json(), parsed_answer=mocked_answer)
    provider = MockLLMProvider(llm_provider_config)

    def side_effect(*args, **kwargs):
        return LLM_response

    provider.async_call_mocker = AsyncMock(side_effect=side_effect)
    return provider


@pytest.fixture
def llm_provider_reasoner_mock(llm_provider_config):
    provider = MockLLMProvider(llm_provider_config)
    answer = LLMResponseType(
        raw_answer='{"reasoning": "The document is very relevant", "score": 2}',
        parsed_answer=RetrievalAnswerEvaluatorFormat(
            reasoning="The document is very relevant",
            score=2,
        ),
    )

    def side_effect(*args, **kwargs):
        return answer

    provider.async_call_mocker = AsyncMock(side_effect=side_effect)
    return provider


class AnswerFormat(BaseModel):
    keyA: str
    keyB: str


class AnswerEvaluatorFormat(BaseModel):
    quality: int
    trustworthiness: int
    originality: int


@pytest.fixture
def answer_format() -> Type[BaseModel]:
    return AnswerFormat


@pytest.fixture
def answer_eval_format() -> Type[BaseModel]:
    return AnswerEvaluatorFormat


@pytest.fixture
def base_answer_eval_config(base_eval_config, answer_eval_format):
    base_config = base_eval_config.model_dump(exclude_unset=True, exclude={"llm_response_schema"})
    base_config["system_prompt"] = "This is a system prompt"
    base_config["user_prompt"] = "Query: {{ query.query }}\nAnswer: {{ answer.text }}"
    base_config["llm_response_schema"] = answer_eval_format
    return BaseAnswerEvaluatorConfig(**base_config)


@pytest.fixture
def pairwise_answer_eval_config(base_answer_eval_config):
    base_config = base_answer_eval_config.model_dump(exclude_unset=True, exclude={"llm_response_schema"})
    base_config["pairwise"] = True
    base_config["evaluator_name"] = AnswerEvaluatorTypes.PAIRWISE
    base_config["user_prompt"] = """
        Query: {{ query.query }}
        [The Start of Assistant A's Answer]
        {{ game.agent_a_answer.text }}
        [The End of Assistant A's Answer]
        [The Start of Assistant B's Answer]
        {{ game.agent_b_answer.text }}
        [The End of Assistant B's Answer]
    """
    return PairwiseEvaluatorConfig(bidirectional=True, **base_config)


@pytest.fixture
def chat_pairwise_answer_eval_config(base_answer_eval_config):
    base_config = base_answer_eval_config.model_dump(
        exclude_unset=True, exclude={"user_prompt", "system_prompt", "llm_response_schema"}
    )
    base_config["pairwise"] = True
    base_config["evaluator_name"] = AnswerEvaluatorTypes.CHAT_PAIRWISE

    return PairwiseEvaluatorConfig(bidirectional=True, **base_config)


@pytest.fixture
def domain_expert_answer_eval_config(base_answer_eval_config):
    base_config = base_answer_eval_config.model_dump(
        exclude_unset=True, exclude={"user_prompt", "system_prompt", "llm_response_schema"}
    )
    base_config["pairwise"] = True
    base_config["expert_in"] = "Computer Science"
    base_config["include_annotations"] = True
    base_config["include_raw_documents"] = True
    base_config["evaluator_name"] = AnswerEvaluatorTypes.DOMAIN_EXPERT
    base_config["include_relevance_reasoning"] = False
    base_config["include_relevance_score"] = False
    return PairwiseDomainExpertEvaluatorConfig(**base_config)


@pytest.fixture
def few_shot_retrieval_eval_config(base_eval_config):
    base_eval_config = base_eval_config.model_dump(exclude_unset=True)
    base_eval_config["evaluator_name"] = RetrievalEvaluatorTypes.FEW_SHOT
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
        few_shots=few_shot_samples,
        **base_eval_config,
    )


@pytest.fixture
def llm_provider_answer_mock(llm_provider_config, answer_eval_format):
    provider = MockLLMProvider(llm_provider_config)
    mocked_answer = LLMResponseType(
        raw_answer='{"quality": 1, "trustworthiness": 0, "originality": 0}',
        parsed_answer=answer_eval_format(quality=1, trustworthiness=0, originality=0),
    )

    def side_effect(*args, **kwargs):
        return mocked_answer

    provider.async_call_mocker = AsyncMock(side_effect=side_effect)
    return provider
