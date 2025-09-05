import json
from unittest.mock import AsyncMock

from pydantic import BaseModel, Field

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
from ragelo.types.answer_formats import RDNAMAnswerEvaluatorFormat, RDNAMAnswerNoAspects
from ragelo.types.formats import LLMInputPrompt, LLMResponseType
from ragelo.types.results import RetrievalEvaluatorResult
from ragelo.utils import string_to_template


class RetrievalEvaluator(BaseRetrievalEvaluator):
    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        return LLMInputPrompt(user_message=f"Query: {query.query}\nDocument: {document.text}")


class TestRetrievalEvaluator:
    def test_evaluate_single_answer(self, llm_provider_mock, experiment, base_retrieval_eval_config):
        base_retrieval_eval_config.llm_response_schema = {"score": "float"}
        evaluator = RetrievalEvaluator.from_config(config=base_retrieval_eval_config, llm_provider=llm_provider_mock)
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        result = evaluator.evaluate(query, doc)
        assert result.raw_answer == '{"score": 1.0}'
        assert result.answer == {"score": 1.0}

        expected_prompt = f"Query: {query.query}\nDocument: {doc.text}"
        call_args = llm_provider_mock.async_call_mocker.call_args_list
        assert call_args[0][0][0].user_message == expected_prompt

    def test_evaluate_experiment(
        self,
        llm_provider_mock,
        experiment,
        base_retrieval_eval_config,
    ):
        base_retrieval_eval_config.llm_response_schema = {"score": "float"}
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
        base_retrieval_eval_config.llm_response_schema = {"score": "float"}
        evaluator = RetrievalEvaluator.from_config(config=base_retrieval_eval_config, llm_provider=llm_provider_mock)
        result = evaluator.evaluate(document="This is a document", query="This is a query")
        call_args = llm_provider_mock.async_call_mocker.call_args_list
        assert call_args[0][0][0].user_message == "Query: This is a query\nDocument: This is a document"
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
            user_prompt="Query: {{ query.query }} Retrieved document: {{ document.text }}",
        )
        assert isinstance(custom_evaluator, BaseRetrievalEvaluator)


class TestRDNAMEvaluator:
    def test_process_single_answer(self, llm_provider_mock_rdnam, rdnam_config, experiment):
        evaluator = RDNAMEvaluator.from_config(config=rdnam_config, llm_provider=llm_provider_mock_rdnam)
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        result = evaluator.evaluate(query, doc)
        assert isinstance(result.answer, RDNAMAnswerEvaluatorFormat)
        assert result.answer.intent_match == 1.0
        assert result.answer.trustworthiness == 0.8
        assert result.answer.overall == 1.2
        assert result.raw_answer is not None
        assert "annotator_1" in result.raw_answer
        assert "annotator_2" in result.raw_answer

        call_args = llm_provider_mock_rdnam.async_call_mocker.call_args_list
        assert call_args[0][0][0].system_prompt.startswith("You are a search quality rater evaluating")


class TestReasonerEvaluator:
    def test_process_single_answer(self, llm_provider_reasoner_mock, base_retrieval_eval_config, experiment):
        evaluator = ReasonerEvaluator.from_config(
            config=base_retrieval_eval_config, llm_provider=llm_provider_reasoner_mock
        )
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        result = evaluator.evaluate(query, doc)
        system_prompt = evaluator.system_prompt.render(query=query, document=doc)
        user_prompt = evaluator.user_prompt.render(query=query, document=doc)
        call_args = llm_provider_reasoner_mock.async_call_mocker.call_args_list

        assert result.answer.score == 2
        assert call_args[0][0][0].user_message == user_prompt
        assert call_args[0][0][0].system_prompt == system_prompt


class TestCustomPromptEvaluator:
    def test_process_single_answer(
        self,
        llm_provider_mock_retrieval,
        custom_prompt_retrieval_eval_config,
        experiment,
    ):
        query = experiment["0"]
        doc = query.retrieved_docs["0"]

        evaluator = CustomPromptEvaluator.from_config(
            config=custom_prompt_retrieval_eval_config,
            llm_provider=llm_provider_mock_retrieval,
        )
        system_prompt = evaluator.system_prompt.render(query=query, document=doc)
        user_prompt = evaluator.user_prompt.render(query=query, document=doc)

        response = evaluator.evaluate(query, doc)
        call_args = llm_provider_mock_retrieval.async_call_mocker.call_args_list

        assert response.answer.score == 2
        assert call_args[0][0][0].user_message == user_prompt
        assert call_args[0][0][0].system_prompt == system_prompt

    def test_process_with_custom_fields(self, llm_provider_mock_retrieval, custom_prompt_retrieval_eval_config):
        custom_prompt_retrieval_eval_config.user_prompt = string_to_template(
            "query: {{ query.query }} doc: {{ document.text }} q_metadata: {{ query.metadata.q_metadata }} d_metadata: {{ document.metadata.d_metadata }}"
        )
        evaluator = get_retrieval_evaluator(
            "custom_prompt",
            llm_provider_mock_retrieval,
            config=custom_prompt_retrieval_eval_config,
        )
        evaluator.evaluate(
            query="this is a query",
            document="this is a document",
            query_metadata={"q_metadata": "q_1"},
            doc_metadata={"d_metadata": "d_1"},
        )
        assert (
            llm_provider_mock_retrieval.async_call_mocker.call_args_list[0][0][0].user_message
            == "query: this is a query doc: this is a document q_metadata: q_1 d_metadata: d_1"
        )


class TestDomainExpertEvaluator:
    def test_process_single_answer(
        self,
        llm_provider_mock_retrieval,
        expert_retrieval_eval_config,
        experiment,
    ):
        evaluator = DomainExpertEvaluator.from_config(
            config=expert_retrieval_eval_config,
            llm_provider=llm_provider_mock_retrieval,
        )
        query = experiment["0"]
        doc = query.retrieved_docs["0"]
        _ = evaluator.evaluate(query, doc)

        assert llm_provider_mock_retrieval.async_call_mocker.call_count == 1
        prompt = llm_provider_mock_retrieval.async_call_mocker.call_args_list[0][0][0]

        assert prompt.system_prompt.startswith("You are a domain expert in")
        assert expert_retrieval_eval_config.expert_in in prompt.system_prompt
        assert expert_retrieval_eval_config.extra_guidelines[0] in prompt.system_prompt
        assert expert_retrieval_eval_config.domain_short in prompt.system_prompt
        assert expert_retrieval_eval_config.company in prompt.system_prompt

        assert prompt.user_message == f"User query:\n{query.query}\n\nDocument passage:\n{doc.text}"


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

        # Check that the call was made with the expected structure
        prompt = call_args[0][0][0]
        system_prompt = prompt.system_prompt
        messages = prompt.messages

        # Check system prompt
        assert system_prompt == "System prompt"
        assert len(messages) == len(few_shot_retrieval_eval_config.few_shots) * 2 + 1

        # Check that user prompt contains few-shot examples and the main query
        assert messages[0]["role"] == messages[2]["role"] == "user"
        assert messages[1]["role"] == messages[3]["role"] == "assistant"
        assert "Few shot query 1" in messages[0]["content"]
        assert "Few shot example 1" in messages[0]["content"]
        assert "Few shot query 2" in messages[2]["content"]
        assert "Few shot example 2" in messages[2]["content"]
        assistant_answer_1 = json.loads(messages[1]["content"])
        assistant_answer_2 = json.loads(messages[3]["content"])
        assert assistant_answer_1["reasoning"] == "This is a good document"
        assert assistant_answer_2["reasoning"] == "This is a bad document"
        assert assistant_answer_1["relevance"] == 2
        assert assistant_answer_2["relevance"] == 0
        assert "What is the capital of Brazil?" in messages[-1]["content"]  # Should contain the actual query


class TestReadmeExamples:
    def test_rdnam_example(self, llm_provider_mock_rdnam):
        def side_effect(*args, **kwargs):
            return LLMResponseType(raw_answer='{"overall": 1.0}', parsed_answer=RDNAMAnswerNoAspects(overall=1))

        llm_provider_mock_rdnam.async_call_mocker = AsyncMock(side_effect=side_effect)
        evaluator = get_retrieval_evaluator("RDNAM", llm_provider=llm_provider_mock_rdnam, write_output=False)
        result = evaluator.evaluate(
            query="What is the capital of France?",
            document="Lyon is the second largest city in France.",
        )
        assert isinstance(result.answer, RDNAMAnswerEvaluatorFormat)
        assert result.answer.overall == 1.0
        assert result.answer == RDNAMAnswerEvaluatorFormat(overall=1.0, intent_match=None, trustworthiness=None)
        assert result.raw_answer == '{"overall": 1.0}'

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

        system_prompt = """You are a helpful assistant for evaluating the relevance of a retrieved document to a user query.
        You should pay extra attention to how **recent** a document is. A document older than 5 years is considered outdated.

        The answer should be evaluated according to its recency, truthfulness, and relevance to the user query.
        """

        user_prompt = """
        User query: {{ query.query }}

        Retrieved document: {{ document.text }}

        The document has a date of {{ document.metadata.date }}.
        Today is {{ query.metadata.today_date }}.
        """

        expected_user_prompt = string_to_template(
            """
        User query: What is the capital of Brazil?

        Retrieved document: Rio de Janeiro is the capital of Brazil.

        The document has a date of 04-03-1950.
        Today is 08-04-2024.
        """
        ).render()

        class ResponseSchema(BaseModel):
            relevance: int = Field(
                description="An integer, either 0 or 1. 0 if the document is irrelevant, 1 if it is relevant."
            )
            recency: int = Field(
                description="An integer, either 0 or 1. 0 if the document is outdated, 1 if it is recent."
            )
            truthfulness: int = Field(
                description="An integer, either 0 or 1. 0 if the document is false, 1 if it is true."
            )
            reasoning: str = Field(
                description="A short explanation of why you think the document is relevant or irrelevant."
            )

        mocked_answer = LLMResponseType(
            raw_answer=json.dumps(answer_dict), parsed_answer=ResponseSchema(**answer_dict)
        )
        evaluator = get_retrieval_evaluator(
            "custom_prompt",  # name of the retrieval evaluator
            llm_provider=llm_provider_mock,  # Which LLM provider to use
            system_prompt=system_prompt,  # your custom prompt
            user_prompt=user_prompt,  # your custom prompt
            llm_response_schema=ResponseSchema,  # The response schema for the LLM.
        )

        def side_effect(*args, **kwargs):
            return mocked_answer

        llm_provider_mock.async_call_mocker = AsyncMock(side_effect=side_effect)

        result = evaluator.evaluate(
            query="What is the capital of Brazil?",  # The user query
            document="Rio de Janeiro is the capital of Brazil.",  # The retrieved document
            query_metadata={"today_date": "08-04-2024"},  # Some metadata for the query
            doc_metadata={"date": "04-03-1950"},  # Some metadata for the document
        )
        call_args = llm_provider_mock.async_call_mocker.call_args_list
        assert result.raw_answer == json.dumps(answer_dict)
        assert result.answer == ResponseSchema(**answer_dict)
        assert call_args[0][0][0].system_prompt == string_to_template(system_prompt).render()
        assert call_args[0][0][0].user_message == expected_user_prompt
