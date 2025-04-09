from __future__ import annotations

import json
from unittest.mock import AsyncMock

from ragelo import get_retrieval_evaluator
from ragelo.evaluators.retrieval_evaluators import (
    BaseRetrievalEvaluator,
    CustomPromptEvaluator,
    DomainExpertEvaluator,
    FewShotEvaluator,
    RDNAMEvaluator,
    ReasonerEvaluator,
)
from ragelo.types import Document, Query
from ragelo.types.formats import LLMResponseType
from ragelo.types.results import RetrievalEvaluatorResult


class RetrievalEvaluator(BaseRetrievalEvaluator):
    def _build_message(self, query: Query, document: Document) -> str:
        return f"Query: {query.query}\nDocument: {document.text}"


class TestRetrievalEvaluator:
    def test_evaluate_single_answer(self, llm_provider_mock, experiment, base_retrieval_eval_config):
        evaluator = RetrievalEvaluator.from_config(config=base_retrieval_eval_config, llm_provider=llm_provider_mock)
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        result = evaluator.evaluate(query, doc)
        assert result.raw_answer == '{"score": 1.0}'
        assert result.answer == {"score": 1.0}

        expected_prompt = f"Query: {query.query}\nDocument: {doc.text}"
        call_args = llm_provider_mock.async_call_mocker.call_args_list
        assert call_args[0][0][0] == expected_prompt

    def test_evaluate_experiment(
        self,
        llm_provider_mock,
        experiment,
        base_retrieval_eval_config,
    ):
        evaluator = RetrievalEvaluator.from_config(config=base_retrieval_eval_config, llm_provider=llm_provider_mock)
        evaluator.evaluate_experiment(experiment)
        doc_ids = ["0", "1", "2", "3"]
        qids = ["0", "0", "1", "1"]
        for query in experiment:
            for doc in query.retrieved_docs_iter():
                assert isinstance(doc.evaluation, RetrievalEvaluatorResult)
                assert isinstance(doc.evaluation.raw_answer, str)
                assert isinstance(doc.evaluation.answer, dict)
                assert doc.did == doc_ids.pop(0)
                assert doc.qid == qids.pop(0)

    def test_evaluate_with_text(self, llm_provider_mock, base_retrieval_eval_config):
        evaluator = RetrievalEvaluator.from_config(config=base_retrieval_eval_config, llm_provider=llm_provider_mock)
        result = evaluator.evaluate(document="This is a document", query="This is a query")
        call_args = llm_provider_mock.async_call_mocker.call_args_list
        assert call_args[0][0][0] == "Query: This is a query\nDocument: This is a document"
        assert isinstance(result.answer, dict)
        assert result.answer["score"] == 1.0

    def test_rich_printing(
        self,
        llm_provider_mock,
        base_eval_config,
        experiment,
        capsys,
    ):
        base_eval_config.rich_print = True
        evaluator = RetrievalEvaluator.from_config(
            config=base_eval_config,
            llm_provider=llm_provider_mock,
        )
        evaluator.evaluate_experiment(experiment)
        captured = capsys.readouterr()
        assert "🔎" in captured.out

    def test_get_by_name(self, llm_provider_mock, expert_retrieval_eval_config):
        domain_expert_evaluator = get_retrieval_evaluator(
            "domain_expert",
            llm_provider_mock,
            expert_in=expert_retrieval_eval_config.expert_in,
        )
        assert isinstance(domain_expert_evaluator, DomainExpertEvaluator)
        reasoner_evaluator = get_retrieval_evaluator(
            "reasoner",
            llm_provider_mock,
        )
        assert isinstance(reasoner_evaluator, ReasonerEvaluator)
        rdna_evaluator = get_retrieval_evaluator("RDNAM", llm_provider_mock)

        assert isinstance(rdna_evaluator, RDNAMEvaluator)
        custom_evaluator = get_retrieval_evaluator(
            "custom_prompt",
            llm_provider_mock,
        )
        assert isinstance(custom_evaluator, BaseRetrievalEvaluator)


class TestRDNAMEvaluator:
    def test_process_single_answer(self, llm_provider_mock_rdnam, rdnam_config, experiment):
        evaluator = RDNAMEvaluator.from_config(config=rdnam_config, llm_provider=llm_provider_mock_rdnam)
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        result = evaluator.evaluate(query, doc)
        assert isinstance(result.answer, dict)
        assert len(result.answer) == 3
        assert result.answer["intent_match"] == 1.5
        assert result.answer["trustworthiness"] == 1.0
        assert result.answer["overall"] == 1.5
        assert result.raw_answer is not None
        assert "annotator_1" in result.raw_answer
        assert "annotator_2" in result.raw_answer

        call_args = llm_provider_mock_rdnam.async_call_mocker.call_args_list
        assert call_args[0][0][0].startswith("You are a search quality rater evaluating")


class TestReasonerEvaluator:
    def test_process_single_answer(self, llm_provider_reasoner_mock, base_retrieval_eval_config, experiment):
        evaluator = ReasonerEvaluator.from_config(
            config=base_retrieval_eval_config, llm_provider=llm_provider_reasoner_mock
        )
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        result = evaluator.evaluate(query, doc)
        formatted_prompt = evaluator.prompt.format(query=query.query, document=doc.text)
        call_args = llm_provider_reasoner_mock.async_call_mocker.call_args_list

        assert result.raw_answer == result.answer
        assert call_args[0][0][0] == formatted_prompt


class TestCustomPromptEvaluator:
    def test_process_single_answer(
        self,
        llm_provider_mock,
        custom_prompt_retrieval_eval_config,
        experiment,
    ):
        query = experiment["0"]
        doc = query.retrieved_docs["0"]

        evaluator = CustomPromptEvaluator.from_config(
            config=custom_prompt_retrieval_eval_config,
            llm_provider=llm_provider_mock,
        )
        formatter = {
            custom_prompt_retrieval_eval_config.query_placeholder: query.query,
            custom_prompt_retrieval_eval_config.document_placeholder: doc.text,
        }
        formatted_prompt = custom_prompt_retrieval_eval_config.prompt.format(**formatter)

        response = evaluator.evaluate(query, doc)
        call_args = llm_provider_mock.async_call_mocker.call_args_list

        assert response.answer == {"score": 1}
        assert call_args[0][0][0] == formatted_prompt

    def test_process_with_custom_fields(self, llm_provider_mock, custom_prompt_retrieval_eval_config):
        custom_prompt_retrieval_eval_config.prompt = (
            "query: {query} doc: {document} q_metadata: {q_metadata} d_metadata: {d_metadata}"
        )
        evaluator = get_retrieval_evaluator(
            "custom_prompt",
            llm_provider_mock,
            config=custom_prompt_retrieval_eval_config,
        )
        evaluator.evaluate(
            query="this is a query",
            document="this is a document",
            query_metadata={"q_metadata": "q_1"},
            doc_metadata={"d_metadata": "d_1"},
        )
        assert (
            llm_provider_mock.async_call_mocker.call_args_list[0][0][0]
            == "query: this is a query doc: this is a document q_metadata: q_1 d_metadata: d_1"
        )


class TestDomainExpertEvaluator:
    def test_sys_prompt(self, llm_provider_mock, expert_retrieval_eval_config):
        evaluator = DomainExpertEvaluator.from_config(
            config=expert_retrieval_eval_config, llm_provider=llm_provider_mock
        )
        assert expert_retrieval_eval_config.expert_in in evaluator.system_prompt
        assert expert_retrieval_eval_config.domain_short in evaluator.system_prompt
        assert f"You work for {expert_retrieval_eval_config.company}" in evaluator.system_prompt

    def test_process_single_answer(
        self,
        llm_provider_mock,
        expert_retrieval_eval_config,
        experiment,
    ):
        evaluator = DomainExpertEvaluator.from_config(
            config=expert_retrieval_eval_config,
            llm_provider=llm_provider_mock,
        )
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        _ = evaluator.evaluate(query, doc)

        assert llm_provider_mock.async_call_mocker.call_count == 2
        call_args = llm_provider_mock.async_call_mocker.call_args_list
        prompts_reasoning = call_args[0][0][0]
        prompts_score = call_args[1][0][0]
        assert len(prompts_reasoning) == 2
        assert len(prompts_score) == 4
        assert prompts_score[0] == prompts_reasoning[0]
        assert prompts_score[1] == prompts_reasoning[1]
        assert prompts_score[0]["role"] == "system"
        assert prompts_score[2]["role"] == "assistant"
        assert prompts_score[1]["role"] == prompts_score[3]["role"] == "user"
        assert prompts_reasoning[0]["content"].startswith("You are a domain expert in")
        assert prompts_score[1]["content"].endswith(expert_retrieval_eval_config.extra_guidelines[0])
        assert prompts_score[3]["content"].startswith("Given the previous reasoning")


class TestFewShotEvaluator:
    def test_process_single_answer(
        self,
        llm_provider_mock,
        few_shot_retrieval_eval_config,
        experiment,
    ):
        evaluator = FewShotEvaluator.from_config(
            config=few_shot_retrieval_eval_config,
            llm_provider=llm_provider_mock,
        )
        query = experiment["0"]
        doc = query.retrieved_docs["0"]

        _ = evaluator.evaluate(query, doc)
        call_args = llm_provider_mock.async_call_mocker.call_args_list
        call_messages = call_args[0][0][0]
        assert len(call_messages) == 6
        assert call_messages[0]["role"] == "system"
        assert call_messages[1]["role"] == call_messages[3]["role"] == call_messages[5]["role"] == "user"
        assert call_messages[2]["role"] == call_messages[4]["role"] == "assistant"


class TestReadmeExamples:
    def test_rdnam_example(self, llm_provider_mock_rdnam):
        def side_effect(*args, **kwargs):
            return LLMResponseType(raw_answer='{"overall": 1}', parsed_answer={"overall": 1})

        llm_provider_mock_rdnam.async_call_mocker = AsyncMock(side_effect=side_effect)
        evaluator = get_retrieval_evaluator("RDNAM", llm_provider=llm_provider_mock_rdnam, write_output=False)
        result = evaluator.evaluate(
            query="What is the capital of France?",
            document="Lyon is the second largest city in France.",
        )
        assert isinstance(result.answer, dict)
        assert result.answer["overall"] == 1
        assert result.answer == {"overall": 1}
        assert result.raw_answer == '{"overall": 1}'

    def test_custom_prompt_evaluator_example(self, llm_provider_mock):
        answer_dict = {
            "relevance": 0,
            "recency": 0,
            "truthfulness": 0,
            "reasoning": (
                "The document is outdated and incorrect. Rio de Janeiro was the capital of "
                "Brazil until 1960 when it was changed to Brasília."
            ),
        }

        prompt = """You are a helpful assistant for evaluating the relevance of a retrieved document to a user query.
You should pay extra attention to how **recent** a document is. A document older than 5 years is considered outdated.

The answer should be evaluated according tot its recency, truthfulness, and relevance to the user query.

User query: {q}

Retrieved document: {d}

The document has a date of {document_date}.
Today is {today_date}.

WRITE YOUR ANSWER ON A SINGLE LINE AS A JSON OBJECT WITH THE FOLLOWING KEYS:
- "relevance": 0 if the document is irrelevant, 1 if it is relevant.
- "recency": 0 if the document is outdated, 1 if it is recent.
- "truthfulness": 0 if the document is false, 1 if it is true.
- "reasoning": A short explanation of why you think the document is relevant or irrelevant.
"""
        response_schema = {
            "relevance": "An integer, either 0 or 1. 0 if the document is irrelevant, 1 if it is relevant.",
            "recency": "An integer, either 0 or 1. 0 if the document is outdated, 1 if it is recent.",
            "truthfulness": "An integer, either 0 or 1. 0 if the document is false, 1 if it is true.",
            "reasoning": "A short explanation of why you think the document is relevant or irrelevant.",
        }
        response_format = "json"

        evaluator = get_retrieval_evaluator(
            "custom_prompt",  # name of the retrieval evaluator
            llm_provider=llm_provider_mock,  # Which LLM provider to use
            prompt=prompt,  # your custom prompt
            query_placeholder="q",  # the placeholder for the query in the prompt
            document_placeholder="d",  # the placeholder for the document in the prompt
            llm_answer_format=response_format,  # The format of the answer. Can be either TEXT, if you expect plain text to be returned, JSON if the answer should be in JSON format, or STRUCTURED, if you provide a Pydantic BaseModel as the response_schema.
            llm_response_schema=response_schema,  # The response schema for the LLM. Required if the llm_answer_format is structured and recommended for JSON.
        )

        def side_effect(*args, **kwargs):
            return LLMResponseType(raw_answer=json.dumps(answer_dict), parsed_answer=answer_dict)

        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=side_effect)

        result = evaluator.evaluate(
            query="What is the capital of Brazil?",  # The user query
            document="Rio de Janeiro is the capital of Brazil.",  # The retrieved document
            query_metadata={"today_date": "08-04-2024"},  # Some metadata for the query
            doc_metadata={"document_date": "04-03-1950"},  # Some metadata for the document
        )
        call_args = llm_provider_mock.async_call_mocker.call_args_list
        assert result.raw_answer == json.dumps(answer_dict)
        assert result.answer == answer_dict
        assert len(call_args) == 1
