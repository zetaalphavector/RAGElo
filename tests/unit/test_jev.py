import asyncio
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
from ragelo.utils import call_async_fn

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


class TestJevBatching:
    """The provider asks prompts that share a batch key, the documents of one query, in one request."""

    @staticmethod
    def provider(requests: list[dict[str, Any]], fail_merged: bool = False, **config: Any) -> VercelJevProvider:
        """Answers each question with the yes-probability written in the document it is about, as `p=0.7`."""

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            requests.append(body)
            if fail_merged and isinstance(body["state"], dict):
                return httpx.Response(400, json={"detail": {"error_type": "max_tokens_exceeded"}})
            answers = {}
            for name in body["questions"]:
                state = body["state"][name.split("__")[0]] if isinstance(body["state"], dict) else body["state"]
                answers[name] = {"type": "boolean", "probability": float(state.split("p=")[1][:3])}
            usage = {"inputTokens": 100 * len(answers) + 1, "outputTokens": 10 * len(answers)}
            return httpx.Response(200, json={"answers": answers, "usage": usage})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return VercelJevProvider(VercelJevConfiguration(api_key=SecretStr("k"), **config), client=client)

    @staticmethod
    def query(qid: str, probabilities: list[float]) -> Query:
        query = Query(qid=qid, query=f"question {qid}")
        for i, probability in enumerate(probabilities):
            query.add_retrieved_doc(Document(qid=qid, did=f"{qid}-d{i}", text=f"document {i} p={probability}"))
        return query

    def test_the_documents_of_a_query_share_a_request_and_keep_their_own_answers(self):
        requests: list[dict[str, Any]] = []
        evaluator = get_retrieval_evaluator("jev", llm_provider=self.provider(requests), n_processes=3)
        query = self.query("q", [0.9, 0.1, 0.5])

        evaluator.evaluate_all_evaluables(query)

        assert len(requests) == 1
        assert requests[0]["state"].keys() == {"item_0", "item_1", "item_2"}
        assert requests[0]["questions"]["item_1__score"]["instructions"].startswith("Consider only `item_1`. ")
        results = [document.evaluations["jev"] for document in query.retrieved_docs.values()]
        assert [result.answer.score for result in results] == [1.8, 0.2, 1.0]
        assert [result.usage.input_tokens for result in results] == [101, 100, 100], "the shares add up to 301"

    def test_batches_are_capped_and_never_mix_queries(self):
        requests: list[dict[str, Any]] = []
        provider = self.provider(requests, batch_size=2)
        evaluator = get_retrieval_evaluator("jev", llm_provider=provider, n_processes=5)
        experiment = Experiment(experiment_name="batches", save_on_disk=False)
        for query in (self.query("a", [0.9, 0.8, 0.7]), self.query("b", [0.1, 0.2])):
            experiment.add_query(query)

        evaluator.evaluate_experiment(experiment)

        sizes = sorted(len(r["state"]) if isinstance(r["state"], dict) else 1 for r in requests)
        assert sizes == [1, 2, 2], "query a is asked as 2 + 1 documents, query b as 2"
        for request in requests:
            if isinstance(request["state"], dict):
                assert len({state.split("\n")[1] for state in request["state"].values()}) == 1, "one query each"

    def test_two_experiments_that_reuse_a_query_id_never_share_a_request(self):
        requests: list[dict[str, Any]] = []
        evaluator = get_retrieval_evaluator("jev", llm_provider=self.provider(requests))
        one, other = self.query("q", [0.9, 0.8]), self.query("q", [0.1, 0.2])

        async def judge_both_at_once() -> None:
            pairs = [(query, document) for query in (one, other) for document in query.retrieved_docs.values()]
            await asyncio.gather(*(evaluator.evaluate_async(pair) for pair in pairs))

        call_async_fn(judge_both_at_once)

        assert len(requests) == 2
        for request in requests:
            probabilities = sorted(float(state.split("p=")[1][:3]) for state in request["state"].values())
            assert probabilities in ([0.8, 0.9], [0.1, 0.2])

    def test_a_batch_size_of_one_asks_every_document_alone(self):
        requests: list[dict[str, Any]] = []
        evaluator = get_retrieval_evaluator("jev", llm_provider=self.provider(requests, batch_size=1), n_processes=3)

        evaluator.evaluate_all_evaluables(self.query("q", [0.9, 0.1, 0.5]))

        assert [type(request["state"]) for request in requests] == [str, str, str]

    def test_a_merged_request_that_fails_is_asked_again_one_document_at_a_time(self):
        requests: list[dict[str, Any]] = []
        evaluator = get_retrieval_evaluator(
            "jev", llm_provider=self.provider(requests, fail_merged=True), n_processes=3
        )
        query = self.query("q", [0.9, 0.1, 0.5])

        evaluator.evaluate_all_evaluables(query)

        assert [type(request["state"]) for request in requests] == [dict, str, str, str]
        assert [d.evaluations["jev"].answer.score for d in query.retrieved_docs.values()] == [1.8, 0.2, 1.0]

    def test_prompts_without_a_batch_key_are_never_merged(self):
        requests: list[dict[str, Any]] = []
        provider = self.provider(requests)
        prompts = [
            LLMInputPrompt(user_message=f"state p=0.{i}", questions={"q": {"type": "boolean", "instructions": "?"}})
            for i in (1, 2, 3)
        ]

        async def ask_all_at_once() -> None:
            await asyncio.gather(*(provider.call_async(prompt, RetrievalEvaluationAnswer) for prompt in prompts))

        call_async_fn(ask_all_at_once)

        assert [type(request["state"]) for request in requests] == [str, str, str]


class TestJevAnswerBatching:
    """The answers and the games of one query share a request, like the documents of one query."""

    @staticmethod
    def provider(requests: list[dict[str, Any]]) -> VercelJevProvider:
        """An answer says how good it is, as `q=2`. It earns that grade, fulfils a criterion from q=1 upwards, and
        beats an answer with a lower one."""

        def quality(text: str, marker: str) -> int:
            return int(text.split(marker, 1)[1].split("q=")[1][0])

        def answer(question: dict[str, Any], state: str) -> dict[str, Any]:
            if question["type"] == "boolean":
                return {"type": "boolean", "probability": 0.9 if quality(state, "[Agent's Report]") >= 1 else 0.1}
            if question["type"] == "score":
                grade = quality(state, "[answer]")
                return {
                    "type": "score",
                    "score": grade,
                    "probabilities": {str(g): float(g == grade) for g in range(3)},
                }
            rubric_game = "Agent A's Answer]" in state
            a = quality(state, "Agent A's Answer]" if rubric_game else "Assistant A]")
            b = quality(state, "Agent B's Answer]" if rubric_game else "Assistant B]")
            winner = "A" if a > b else "B" if b > a else "C"
            return {"type": "choice", "choice": winner, "probabilities": {w: float(w == winner) for w in "ABC"}}

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            requests.append(body)
            answers = {}
            for name, question in body["questions"].items():
                state = body["state"][name.split("__")[0]] if isinstance(body["state"], dict) else body["state"]
                answers[name] = answer(question, state)
            return httpx.Response(200, json={"answers": answers, "usage": {"inputTokens": 300, "outputTokens": 30}})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return VercelJevProvider(VercelJevConfiguration(api_key=SecretStr("k")), client=client)

    @staticmethod
    def experiment() -> Experiment:
        experiment = Experiment(experiment_name="answers", save_on_disk=False)
        experiment.add_query(Query(qid="q", query="capital of France?"))
        for agent, quality in (("poor", 0), ("fair", 1), ("good", 2)):
            experiment.add_agent_answer(f"An answer of q={quality}", agent, "q")
        return experiment

    def test_the_answers_of_a_query_share_a_request_and_keep_their_own_scores(self):
        requests: list[dict[str, Any]] = []
        experiment = self.experiment()

        get_answer_evaluator("jev", llm_provider=self.provider(requests), n_processes=3).evaluate_experiment(
            experiment
        )

        assert [len(request["state"]) for request in requests] == [3]
        scores = {agent: answer.evaluations["jev"].answer.score for agent, answer in experiment["q"].answers.items()}
        assert scores == {"poor": 0, "fair": 1, "good": 2}

    def test_the_games_of_a_query_share_one_request_per_answer_order(self):
        requests: list[dict[str, Any]] = []
        experiment = self.experiment()
        evaluator = get_answer_evaluator("jev_pairwise", llm_provider=self.provider(requests), n_processes=3)

        evaluator.evaluate_experiment(experiment)

        assert [len(request["state"]) for request in requests] == [3, 3], "the three games, forward then reversed"
        winners = {
            (game.agent_a_answer.agent, game.agent_b_answer.agent): {
                "A": game.agent_a_answer.agent,
                "B": game.agent_b_answer.agent,
            }[game.evaluations["jev_pairwise"].answer.winner]
            for game in experiment["q"].pairwise_games.values()
        }
        assert winners == {("fair", "good"): "good", ("fair", "poor"): "fair", ("good", "poor"): "good"}

    def test_rubric_criteria_of_every_answer_go_in_one_request(self):
        requests: list[dict[str, Any]] = []
        experiment = self.experiment()
        experiment["q"].rubric = [
            Criterion(criterion_name="names_capital", short_question="Does the response name the capital?"),
            Criterion(criterion_name="is_short", short_question="Is the response short?"),
        ]
        evaluator = get_answer_evaluator("jev_rubric_pointwise", llm_provider=self.provider(requests), n_processes=3)

        evaluator.evaluate_experiment(experiment)

        assert len(requests) == 1
        assert len(requests[0]["questions"]) == 6
        assert requests[0]["questions"]["item_0__is_short"]["instructions"] == (
            "Consider only `item_0`. Is the response short?"
        )
        fulfilled = {
            agent: [c.fulfillment for c in answer.evaluations["jev_rubric_pointwise"].answer.criteria]
            for agent, answer in experiment["q"].answers.items()
        }
        assert fulfilled == {"poor": [False, False], "fair": [True, True], "good": [True, True]}

    def test_rubric_games_of_a_query_share_one_request_per_answer_order(self):
        requests: list[dict[str, Any]] = []
        experiment = self.experiment()
        experiment["q"].rubric = [Criterion(criterion_name="names_capital", short_question="Names the capital?")]
        evaluator = get_answer_evaluator("jev_rubric_pairwise", llm_provider=self.provider(requests), n_processes=3)

        evaluator.evaluate_experiment(experiment)

        assert [len(request["state"]) for request in requests] == [3, 3]
        winners = {
            (game.agent_a_answer.agent, game.agent_b_answer.agent): game.evaluations[
                "jev_rubric_pairwise"
            ].answer.winner
            for game in experiment["q"].pairwise_games.values()
        }
        assert winners == {("fair", "good"): "B", ("fair", "poor"): "A", ("good", "poor"): "A"}
