import json
import os

import pytest
from pydantic import SecretStr

from ragelo import AgentAnswer, Query, get_answer_evaluator, get_llm_provider, get_retrieval_evaluator
from ragelo.types import LLMInputPrompt
from ragelo.types.answer_formats import RetrievalEvaluationAnswer
from ragelo.types.configurations import TypeSafeConfiguration
from ragelo.types.formats import JevResponse, LLMUsage

httpx2 = pytest.importorskip("httpx2")
typesafe_sdk = pytest.importorskip("typesafe_sdk")

from ragelo.llm_providers import TypeSafeProvider

ANSWERS = {
    "relevant": {"type": "noul", "noul": 0.9},
    "score": {
        "type": "score",
        "score": 1.6,
        "confidence": 0.7,
        "legend": {"0": "bad", "1": "fine", "2": "good"},
        "probabilities": {"0": 0.0, "1": 0.4, "2": 0.6},
    },
    "winner": {"type": "choice", "choice": "B", "confidence": 0.8, "probabilities": {"A": 0.1, "B": 0.9}},
}


def typesafe_provider(requests: list, status: int = 200) -> TypeSafeProvider:
    """A TypeSafeProvider over the real SDK client, whose API answers every question asked from `ANSWERS`."""

    def handler(request):
        requests.append(request)
        asked = json.loads(request.content)["questions"]
        body = {
            "model": "jev-latest",
            "answers": {name: ANSWERS[name] for name in asked},
            "usage": {"input_tokens": 200, "output_tokens": 20},
        }
        return httpx2.Response(status, json=body if status == 200 else {"error": "no credits"})

    client = typesafe_sdk.AsyncTypeSafeClient(
        api_key="typesafe-key",
        transport=httpx2.MockTransport(handler),
        retry=typesafe_sdk.RetryPolicy(max_retries=0),
    )
    return TypeSafeProvider(TypeSafeConfiguration(api_key=SecretStr("typesafe-key")), client=client)


class TestTypeSafeProvider:
    def test_asks_the_sdk_the_jev_questions_and_returns_the_shared_jev_response(self):
        requests: list = []
        prompt = LLMInputPrompt(
            system_prompt="Is the document relevant?",
            user_message="the state",
            questions={
                "relevant": {"type": "boolean"},
                "score": {"type": "score", "criteria": ["bad", "fine", "good"]},
                "winner": {"type": "choice", "instructions": "Which is better?", "criteria": {"A": "a", "B": "b"}},
            },
        )

        result = typesafe_provider(requests)(prompt, response_schema=RetrievalEvaluationAnswer)

        sent = json.loads(requests[0].content)
        assert requests[0].headers["authorization"] == "Bearer typesafe-key"
        assert (sent["state"], sent["model"]) == ("the state", "jev-latest")
        assert sent["questions"]["relevant"] == {"type": "noul", "instructions": "Is the document relevant?"}
        assert sent["questions"]["winner"]["instructions"] == "Which is better?"
        response = result.parsed_answer
        assert isinstance(response, JevResponse)
        assert (response.answers["relevant"].type, response.answers["relevant"].probability) == ("boolean", 0.9)
        assert response.answers["score"].probabilities == {"0": 0.0, "1": 0.4, "2": 0.6}
        assert (response.answers["score"].label, response.answers["score"].confidence) == ("2", 0.7)
        assert response.answers["winner"].label == "B"
        assert result.usage == LLMUsage(input_tokens=200, output_tokens=20)

    def test_an_api_error_names_typesafe_and_the_status(self):
        prompt = LLMInputPrompt(user_message="state", questions={"relevant": {"type": "boolean", "instructions": "?"}})
        with pytest.raises(ValueError, match="TypeSafe request failed.*403"):
            typesafe_provider([], status=403)(prompt, response_schema=RetrievalEvaluationAnswer)

    def test_the_factory_reads_the_typesafe_key(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-key")
        provider = get_llm_provider("typesafe")
        assert isinstance(provider, TypeSafeProvider)
        assert provider.config.api_key.get_secret_value() == "typesafe-key"

    def test_the_jev_evaluators_accept_it(self):
        requests: list = []
        evaluator = get_retrieval_evaluator("jev", llm_provider=typesafe_provider(requests), boolean_question=False)

        result = evaluator.evaluate(query="capital of France?", document="Paris is the capital of France.")

        assert (result.answer.score, result.answer.confidence) == (2, 0.7)
        assert json.loads(requests[0].content)["questions"]["score"]["criteria"] == list(evaluator.relevance_grades)

    @pytest.mark.requires_typesafe
    def test_typesafe_integration(self):
        """Real calls through the TypeSafe API. Requires TYPESAFE_API_KEY. Run with: pytest --runtypesafe"""
        provider = get_llm_provider("typesafe", api_key=os.environ["TYPESAFE_API_KEY"])
        query = Query(qid="q", query="What is the capital of France?")

        relevant = get_retrieval_evaluator("jev", llm_provider=provider).evaluate(
            query=query, document="Paris is the capital and largest city of France."
        )
        off_topic = get_retrieval_evaluator("jev", llm_provider=provider, boolean_question=False).evaluate(
            query=query, document="Bananas are rich in potassium."
        )
        game = get_answer_evaluator("jev_pairwise", llm_provider=provider).evaluate(
            query=query,
            answer_a=AgentAnswer(qid="q", agent="a", text="Lyon, I think."),
            answer_b=AgentAnswer(qid="q", agent="b", text="The capital of France is Paris."),
        )

        assert relevant.answer.score > 1.5
        assert (off_topic.answer.score, set(off_topic.answer.probabilities)) == (0, {"0", "1", "2"})
        assert game.answer.winner == "B"
        assert relevant.usage.input_tokens > 0
