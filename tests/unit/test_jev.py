import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from ragelo import Experiment, get_answer_evaluator, get_retrieval_evaluator
from ragelo.evaluators.jev_evaluator_mixin import JEV_REASONING
from ragelo.evaluators.retrieval_evaluators import RDNAMEvaluator
from ragelo.llm_providers import VercelJevProvider
from ragelo.types import Document, LLMInputPrompt, PairwiseGame, Query
from ragelo.types.answer_formats import Criterion, RDNAMEvaluationAnswer, RetrievalEvaluationAnswer
from ragelo.types.configurations import VercelJevConfiguration
from ragelo.types.formats import JevResponse, LLMUsage

Answerer = Callable[[str, dict[str, Any], str], dict[str, Any]]


def jev_provider(answerer: Answerer, requests: list[httpx.Request] | None = None) -> VercelJevProvider:
    """A VercelJevProvider whose gateway answers each question with `answerer(name, question, state)`."""

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        body = json.loads(request.content)
        answers = {name: answerer(name, question, body["state"]) for name, question in body["questions"].items()}
        confidence = dict.fromkeys(answers, 0.8)
        return httpx.Response(
            200,
            json={
                "answers": answers,
                "usage": {"inputTokens": 278, "outputTokens": 20},
                "providerMetadata": {"typesafe": {"confidence": confidence}},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return VercelJevProvider(VercelJevConfiguration(api_key=SecretStr("gateway-key")), client=client)


def relevant(score: int) -> RetrievalEvaluationAnswer:
    return RetrievalEvaluationAnswer(reasoning="Names the capital.", score=score)


def score_answer(probabilities: dict[str, float]) -> dict[str, Any]:
    mean = sum(int(level) * p for level, p in probabilities.items())
    return {"type": "score", "score": mean, "probabilities": probabilities}


class TestVercelJevProvider:
    def test_sends_the_state_and_the_questions_with_the_system_prompt_as_instructions(self):
        requests: list[httpx.Request] = []
        provider = jev_provider(lambda *_: score_answer({"0": 0.1, "1": 0.9}), requests)
        prompt = LLMInputPrompt(
            system_prompt="Is it relevant?",
            user_message="the state",
            questions={"score": {"type": "score", "criteria": ["no", "yes"]}},
        )

        response = provider(prompt, response_schema=RetrievalEvaluationAnswer).parsed_answer

        request = requests[0]
        assert request.headers["authorization"] == "Bearer gateway-key"
        assert request.headers["ai-model-id"] == "typesafe-ai/jev"
        assert json.loads(request.content) == {
            "state": "the state",
            "questions": {"score": {"type": "score", "instructions": "Is it relevant?", "criteria": ["no", "yes"]}},
        }
        assert isinstance(response, JevResponse)
        assert response.answers["score"].label == "1"
        assert response.answers["score"].confidence == 0.8

    def test_a_prompt_without_questions_is_rejected(self):
        provider = jev_provider(lambda *_: {})
        with pytest.raises(ValueError, match="typed questions"):
            provider(LLMInputPrompt(user_message="Write a poem"), response_schema=RetrievalEvaluationAnswer)

    def test_a_gateway_error_carries_the_status_and_body(self):
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(403, text="no credits")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = VercelJevProvider(VercelJevConfiguration(api_key=SecretStr("k")), client=client)
        prompt = LLMInputPrompt(user_message="state", questions={"q": {"type": "boolean", "instructions": "?"}})
        with pytest.raises(ValueError, match="403 no credits"):
            provider(prompt, response_schema=RetrievalEvaluationAnswer)
        assert len(requests) == 1, "a status that a second try cannot fix is not retried"

    @pytest.mark.parametrize(
        "body", ['{"result": []}', '{"answers": {"q": {"type": "essay"}}}', "<html>bad gateway</html>"]
    )
    def test_an_unexpected_answer_format_says_what_the_gateway_sent(self, body):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=body)))
        provider = VercelJevProvider(VercelJevConfiguration(api_key=SecretStr("k")), client=client)
        prompt = LLMInputPrompt(user_message="state", questions={"q": {"type": "boolean", "instructions": "?"}})
        with pytest.raises(ValueError, match="unexpected format") as error:
            provider(prompt, response_schema=RetrievalEvaluationAnswer)
        assert body in str(error.value)

    def test_a_confidence_inside_the_answer_is_kept(self):
        answer = {"type": "choice", "choice": "A", "probabilities": {"A": 0.9, "B": 0.1}, "confidence": 0.7}
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"answers": {"q": answer}}))
        )
        provider = VercelJevProvider(VercelJevConfiguration(api_key=SecretStr("k")), client=client)
        prompt = LLMInputPrompt(user_message="state", questions={"q": {"type": "choice", "instructions": "?"}})
        assert provider(prompt, response_schema=RetrievalEvaluationAnswer).parsed_answer.answers["q"].confidence == 0.7

    @pytest.mark.parametrize(
        ("failure", "failures", "raised"),
        [
            (503, 2, None),
            (503, 3, ValueError),
            (429, 2, None),
            (502, 2, None),
            ("timeout", 2, None),
            ("timeout", 3, httpx.ReadTimeout),
            ("connect", 2, None),
        ],
    )
    def test_a_passing_failure_is_retried_with_backoff_until_the_retries_run_out(self, failure, failures, raised):
        statuses = [failure] * failures + [200]
        delays = [0.5, 1.0]
        slept: list[float] = []

        def handler(_: httpx.Request) -> httpx.Response:
            status = statuses.pop(0)
            if status == "timeout":
                raise httpx.ReadTimeout("")
            if status == "connect":
                raise httpx.ConnectError("")
            return httpx.Response(status, json={"answers": {"q": {"type": "boolean", "probability": 0.9}}})

        async def sleep(delay: float) -> None:
            slept.append(delay)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        provider = VercelJevProvider(VercelJevConfiguration(api_key=SecretStr("k")), client=client, sleep=sleep)
        prompt = LLMInputPrompt(user_message="state", questions={"q": {"type": "boolean"}})

        if raised is None:
            response = provider(prompt, response_schema=RetrievalEvaluationAnswer).parsed_answer
            assert response.answers["q"].probability == 0.9
        else:
            with pytest.raises(raised):
                provider(prompt, response_schema=RetrievalEvaluationAnswer)
        assert slept == delays


class TestJevEvaluators:
    def test_only_the_jev_provider_is_accepted(self, llm_provider_mock):
        with pytest.raises(TypeError, match='"typesafe" and "vercel-jev" LLM providers'):
            get_retrieval_evaluator("jev", llm_provider=llm_provider_mock)

    def test_answer_evaluators_are_told_what_jev_found_about_a_document(self, experiment: Experiment):
        """Pairwise prompts quote each document's relevance reasoning by default."""
        provider = jev_provider(lambda *_: {"type": "boolean", "probability": 0.9})
        get_retrieval_evaluator("jev", llm_provider=provider).evaluate_experiment(experiment)
        query = experiment["0"]
        game = PairwiseGame(qid="0", agent_a_answer=query.answers["agent1"], agent_b_answer=query.answers["agent2"])

        prompt = get_answer_evaluator("jev_pairwise", llm_provider=provider)._build_message_pairwise(query, game)

        assert JEV_REASONING not in prompt.user_message
        assert "[0]  Jev gives a probability of 0.90 that the document is relevant." in prompt.user_message

    def test_score_question_takes_the_most_likely_grade_not_the_mean(self, experiment: Experiment):
        """A 0.4 / 0.0 / 0.6 split has a mean of 1.2, which no single grade got any probability for."""
        requests: list[httpx.Request] = []
        provider = jev_provider(lambda *_: score_answer({"0": 0.4, "1": 0.0, "2": 0.6}), requests)
        evaluator = get_retrieval_evaluator("jev", llm_provider=provider, boolean_question=False)

        evaluator.evaluate_experiment(experiment)

        answer = experiment["0"].retrieved_docs["0"].evaluations["jev"].answer
        assert answer.score == 2
        assert answer.probabilities == {"0": 0.4, "1": 0.0, "2": 0.6}
        assert answer.confidence == 0.8
        assert answer.reasoning == f"Jev's most likely relevance grade: {evaluator.relevance_grades[2]}"
        assert experiment.get_qrels()["0"] == {"0": 2, "1": 2}
        question = json.loads(requests[0].content)["questions"]["score"]
        assert question["criteria"] == list(evaluator.relevance_grades)

    def test_retrieval_scales_the_yes_probability_to_the_top_grade_by_default(self, experiment: Experiment):
        requests: list[httpx.Request] = []
        provider = jev_provider(lambda *_: {"type": "boolean", "probability": 0.8}, requests)
        evaluator = get_retrieval_evaluator("jev", llm_provider=provider)

        evaluator.evaluate_experiment(experiment)

        question = json.loads(requests[0].content)["questions"]["score"]
        assert question == {"type": "boolean", "instructions": evaluator.boolean_prompt}
        assert experiment["0"].retrieved_docs["0"].evaluations["jev"].answer.score == pytest.approx(1.6)
        assert experiment.get_qrels()["0"] == {"0": 2, "1": 2}

    @pytest.mark.parametrize(
        ("evaluator_name", "kwargs"), [("jev", {}), ("jev_rubric_coverage", {"expert_in": "geography"})]
    )
    def test_documents_longer_than_jev_accepts_are_cut_to_their_start(self, evaluator_name, kwargs, caplog):
        requests: list[httpx.Request] = []
        provider = jev_provider(lambda *_: {"type": "boolean", "probability": 0.8}, requests)
        evaluator = get_retrieval_evaluator(evaluator_name, llm_provider=provider, max_document_chars=20, **kwargs)
        query = Query(
            qid="q", query="capital of France?", rubric=[Criterion(criterion_name="names_capital", short_question="?")]
        )
        long_document = Document(qid="q", did="long", text="Paris is the capital " + "and so on. " * 50)
        short_document = Document(qid="q", did="short", text="Paris.")

        evaluator.evaluate(query, long_document)
        evaluator.evaluate(query, short_document)

        long_state, short_state = (json.loads(request.content)["state"] for request in requests)
        assert "Paris is the capital" in long_state and "and so on" not in long_state
        assert "Paris." in short_state
        assert "Document long cut from" in caplog.text and "Document short" not in caplog.text

    def test_a_system_prompt_replaces_the_yes_no_question(self):
        requests: list[httpx.Request] = []
        provider = jev_provider(lambda *_: {"type": "boolean", "probability": 0.25}, requests)
        evaluator = get_retrieval_evaluator(
            "jev", llm_provider=provider, system_prompt="Would you cite the document when answering {{ query.query }}?"
        )

        result = evaluator.evaluate(query="capital of France?", document="Paris is the capital of France.")

        question = json.loads(requests[0].content)["questions"]["score"]
        assert question == {
            "type": "boolean",
            "instructions": "Would you cite the document when answering capital of France??",
        }
        assert result.answer.score == pytest.approx(0.5)

    def test_custom_relevance_grades_become_the_score_levels(self):
        requests: list[httpx.Request] = []
        provider = jev_provider(lambda *_: score_answer({"0": 0.0, "1": 0.0, "2": 0.1, "3": 0.9}), requests)
        grades = ["off topic", "on topic", "partial answer", "full answer"]
        evaluator = get_retrieval_evaluator(
            "jev", llm_provider=provider, relevance_grades=grades, boolean_question=False
        )

        result = evaluator.evaluate(query="capital of France?", document="Paris is the capital of France.")

        assert json.loads(requests[0].content)["questions"]["score"]["criteria"] == grades
        assert result.answer.score == 3

    def test_pointwise_answer_score(self, experiment: Experiment):
        provider = jev_provider(lambda *_: score_answer({"0": 0.1, "1": 0.7, "2": 0.2}))
        evaluator = get_answer_evaluator("jev", llm_provider=provider)

        evaluator.evaluate_experiment(experiment)

        answer = experiment["0"].answers["agent1"].evaluations["jev"].answer
        assert (answer.score, answer.reasoning) == (1, JEV_REASONING)

    @pytest.mark.parametrize(
        ("forward", "reversed_order", "winner", "probabilities"),
        [
            ({"A": 0.6, "B": 0.3, "C": 0.1}, {"A": 0.2, "B": 0.7, "C": 0.1}, "A", {"A": 0.65, "B": 0.25, "C": 0.1}),
            ({"A": 0.9, "B": 0.1, "C": 0.0}, {"A": 0.9, "B": 0.1, "C": 0.0}, "C", {"A": 0.5, "B": 0.5, "C": 0.0}),
            ({"A": 0.2, "B": 0.1, "C": 0.7}, {"A": 0.1, "B": 0.2, "C": 0.7}, "C", {"A": 0.2, "B": 0.1, "C": 0.7}),
        ],
        ids=["agreement", "position bias cancels to a tie", "tie is the most likely outcome"],
    )
    def test_pairwise_averages_both_answer_orders(
        self, experiment: Experiment, forward, reversed_order, winner, probabilities
    ):
        query = experiment["0"]
        first = query.answers["agent1"].text

        def answerer(name: str, question: dict[str, Any], state: str) -> dict[str, Any]:
            asked_forward = state.index(first) < state.index(query.answers["agent2"].text)
            distribution = forward if asked_forward else reversed_order
            return {"type": "choice", "choice": max(distribution, key=distribution.get), "probabilities": distribution}

        evaluator = get_answer_evaluator("jev_pairwise", llm_provider=jev_provider(answerer))
        game = PairwiseGame(qid="0", agent_a_answer=query.answers["agent1"], agent_b_answer=query.answers["agent2"])

        result = evaluator.evaluate(query=query, answer_a=game.agent_a_answer, answer_b=game.agent_b_answer)

        assert result.answer.winner == winner
        assert result.answer.probabilities == pytest.approx(probabilities)

    @pytest.mark.parametrize(
        ("options", "shown", "hidden"),
        [
            ({}, ["[0]  Names the capital."], ["Content: Brasília", '"']),
            ({"include_relevance_reasoning": False}, [], ["[Reference Documents]"]),
            ({"include_relevance_reasoning": False, "include_relevance_score": True}, ["[0]  2 "], ["Names the"]),
            (
                {"include_raw_documents": True},
                ["Content: Brasília is the capital of Brazil.", "Relevance:  Names the capital.", "[1]: Rio de"],
                [],
            ),
        ],
    )
    def test_pairwise_state_follows_the_document_options_of_the_pairwise_evaluator(
        self, experiment: Experiment, retrieval_evaluation, options, shown, hidden
    ):
        """The options are inherited from the pairwise config, so the state has to honour them too."""
        requests: list[httpx.Request] = []
        provider = jev_provider(
            lambda *_: {"type": "choice", "probabilities": {"A": 0.6, "B": 0.3, "C": 0.1}}, requests
        )
        query = experiment["0"]
        experiment.add_evaluation(
            (query, query.retrieved_docs["0"]), retrieval_evaluation.model_copy(update={"answer": relevant(2)})
        )
        evaluator = get_answer_evaluator("jev_pairwise", llm_provider=provider, **options)

        evaluator.evaluate(query=query, answer_a=query.answers["agent1"], answer_b=query.answers["agent2"])

        request = json.loads(requests[0].content)
        assert request["questions"]["winner"]["instructions"].startswith("Which assistant gave the better answer")
        assert query.answers["agent1"].text in request["state"] and query.answers["agent2"].text in request["state"]
        assert all(text in request["state"] for text in shown)
        assert not any(text in request["state"] for text in hidden)

    def test_rubric_pointwise_asks_every_criterion_in_one_request(self, experiment: Experiment):
        requests: list[httpx.Request] = []
        yes_probability = {"names_capital": 0.9, "gives_population": 0.2}
        provider = jev_provider(lambda name, *_: {"type": "boolean", "probability": yes_probability[name]}, requests)
        query = experiment["0"]
        query.rubric = [
            Criterion(criterion_name="names_capital", short_question="Does the response name the capital?"),
            Criterion(criterion_name="gives_population", short_question="Does the response give the population?"),
        ]
        evaluator = get_answer_evaluator("jev_rubric_pointwise", llm_provider=provider, expert_in="geography")

        result = evaluator.evaluate(query=query, answer=query.answers["agent1"])

        assert len(requests) == 1
        questions = json.loads(requests[0].content)["questions"]
        assert questions["names_capital"] == {"type": "boolean", "instructions": "Does the response name the capital?"}
        assert {c.criterion.criterion_name: (c.fulfillment, c.probability) for c in result.answer.criteria} == {
            "names_capital": (True, 0.9),
            "gives_population": (False, 0.2),
        }

    def test_rubric_options_that_need_generated_text_are_rejected(self):
        provider = jev_provider(lambda *_: {})
        with pytest.raises(ValueError, match="graduated_scoring"):
            get_answer_evaluator(
                "jev_rubric_pointwise", llm_provider=provider, expert_in="geography", graduated_scoring=True
            )

    def test_probabilities_survive_a_save_and_reload(self, tmp_path, base_experiment_config):
        save_path = tmp_path / "jev.json"
        config = {**base_experiment_config, "save_on_disk": True, "save_path": str(save_path)}
        experiment = Experiment(**config)
        provider = jev_provider(lambda *_: score_answer({"0": 0.0, "1": 0.25, "2": 0.75}))
        get_retrieval_evaluator("jev", llm_provider=provider, boolean_question=False).evaluate_experiment(experiment)
        experiment.save()

        reloaded = Experiment(**config)

        answer = reloaded["0"].retrieved_docs["0"].evaluations["jev"].answer
        assert (answer.score, answer.probabilities) == (2, {"0": 0.0, "1": 0.25, "2": 0.75})
        assert reloaded["0"].retrieved_docs["0"].evaluations["jev"].usage == LLMUsage(
            input_tokens=278, output_tokens=20
        )


class TestJevRDNAMEvaluator:
    def test_the_relevance_is_one_score_question_over_the_rdnam_grades(self, base_experiment_config, tmp_path):
        requests: list[httpx.Request] = []
        provider = jev_provider(lambda *_: score_answer({"0": 0.0, "1": 0.46, "2": 0.54}), requests)
        config = base_experiment_config | {"save_on_disk": True, "save_path": str(tmp_path / "jev_rdnam.json")}
        evaluator = get_retrieval_evaluator("jev_rdnam", llm_provider=provider)

        evaluator.evaluate_experiment(Experiment(**config))

        questions = json.loads(requests[0].content)["questions"]
        assert questions.keys() == {"score"}
        assert questions["score"]["criteria"] == list(RDNAMEvaluator.relevance_grades)
        assert "Assume that you are writing a report" in questions["score"]["instructions"]
        reloaded = Experiment(**config)
        answer = reloaded["0"].retrieved_docs["0"].evaluations["jev_rdnam"].answer
        assert isinstance(answer, RDNAMEvaluationAnswer)
        assert answer.score == pytest.approx(1.54)
        assert (answer.probabilities, answer.confidence) == ({"0": 0.0, "1": 0.46, "2": 0.54}, 0.8)
        assert answer.intent_match is None
        assert answer.reasoning == "Jev's expected relevance grade is 1.54 out of 2."
        assert reloaded.get_qrels()["0"]["0"] == 2

    def test_aspects_are_extra_questions(self):
        requests: list[httpx.Request] = []
        means = {"score": 1.2, "intent_match": 2.0, "trustworthiness": 1.0}
        provider = jev_provider(
            lambda name, *_: {"type": "score", "score": means[name], "probabilities": {}}, requests
        )
        evaluator = get_retrieval_evaluator("jev_rdnam", llm_provider=provider, use_aspects=True)

        result = evaluator.evaluate(query="capital of France?", document="Paris is the capital of France.")

        questions = json.loads(requests[0].content)["questions"]
        assert questions.keys() == {"intent_match", "trustworthiness", "score"}
        assert questions["intent_match"]["criteria"] == ["0 out of 2", "1 out of 2", "2 out of 2"]
        assert (result.answer.score, result.answer.intent_match, result.answer.trustworthiness) == (1.2, 2.0, 1.0)

    def test_several_annotators_are_rejected(self):
        """Jev repeats itself, so five annotators would be five identical judgments."""
        with pytest.raises(ValueError, match="use_multiple_annotators"):
            get_retrieval_evaluator(
                "jev_rdnam", llm_provider=jev_provider(lambda *_: {}), use_multiple_annotators=True
            )
