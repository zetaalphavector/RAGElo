from ragelo import get_retrieval_evaluator
from ragelo.evaluators.retrieval_evaluators import (
    BaseRetrievalEvaluator,
    CustomPromptEvaluator,
    DomainExpertEvaluator,
    FewShotEvaluator,
    RDNAMEvaluator,
    ReasonerEvaluator,
)
from ragelo.types import Document, EvaluatorAnswer, Query


class RetrievalEvaluator(BaseRetrievalEvaluator):
    def _build_message(self, query: Query, document: Document) -> str:
        return f"Query: {query.query}\nDocument: {document.text}"

    def _process_answer(self, answer: str) -> str:
        return self.json_answer_parser(answer, key="relevance")


class TestRetrievalEvaluator:
    def test_evaluate_single_answer(
        self,
        llm_provider_json_mock,
        retrieval_eval_config,
        qs_with_docs,
    ):
        evaluator = RetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_json_mock
        )
        query = qs_with_docs[0]
        doc = query.retrieved_docs[0]
        raw_answer, answer = evaluator.evaluate(query, doc)
        assert raw_answer == 'LLM JSON response\n{"relevance": 0}'
        assert answer == 0

        expected_prompt = f"Query: {query.query}\nDocument: {doc.text}"
        call_args = llm_provider_json_mock.call_mocker.call_args_list
        assert call_args[0][0][0] == expected_prompt

    def test_batch_eval(
        self,
        llm_provider_json_mock,
        retrieval_eval_config,
        qs_with_docs,
    ):
        evaluator = RetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_json_mock
        )
        results = evaluator.batch_evaluate(qs_with_docs)
        assert len(results) == 4
        assert results[0].qid == results[1].qid == "0"
        assert results[2].qid == results[3].qid == "1"
        assert results[0].did == "0"
        assert results[2].did == "2"

        call_args = llm_provider_json_mock.call_mocker.call_args_list
        expected_prompts = []
        for query in qs_with_docs:
            for doc in query.retrieved_docs:
                expected_prompts.append(f"Query: {query.query}\nDocument: {doc.text}")
        for i, call in enumerate(call_args):
            assert call[0][0] == expected_prompts[i]

    def test_evaluate_with_text(self, llm_provider_json_mock, retrieval_eval_config):
        evaluator = RetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_json_mock
        )
        raw_answer, processed_answer = evaluator.evaluate(
            document="This is a query", query="This is a document"
        )
        assert raw_answer == 'LLM JSON response\n{"relevance": 0}'
        assert processed_answer == 0

    def test_rich_printing(
        self,
        llm_provider_json_mock,
        retrieval_eval_config,
        qs_with_docs,
        capsys,
    ):
        retrieval_eval_config.rich_print = True
        evaluator = RetrievalEvaluator.from_config(
            config=retrieval_eval_config,
            llm_provider=llm_provider_json_mock,
        )
        _ = evaluator.batch_evaluate(qs_with_docs)
        captured = capsys.readouterr()
        assert "🔎" in captured.out

    def test_get_by_name(self, llm_provider_mock, expert_retrieval_eval_config):
        domain_expert_evaluator = get_retrieval_evaluator(
            "domain_expert",
            llm_provider_mock,
            domain_long=expert_retrieval_eval_config.domain_long,
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
            output_file="tests/data/output.csv",
        )
        assert isinstance(custom_evaluator, BaseRetrievalEvaluator)


class TestRDNAMEvaluator:
    def test_process_single_answer(
        self, llm_provider_mock_rdnam, rdnam_config, qs_with_docs
    ):
        evaluator = RDNAMEvaluator.from_config(
            config=rdnam_config, llm_provider=llm_provider_mock_rdnam
        )
        results = evaluator.evaluate(qs_with_docs["0"]["0"])
        assert isinstance(results, dict)
        assert results["answer"] == 1
        assert results["query_id"] == "0"
        assert results["did"] == "0"
        call_args = llm_provider_mock_rdnam.call_mocker.call_args_list
        assert call_args[0][0][0].startswith(
            "You are a search quality rater evaluating"
        )


class TestReasonerEvaluator:
    def test_process_single_answer(
        self,
        llm_provider_mock,
        retrieval_eval_config,
        qs_with_docs,
    ):
        evaluator = ReasonerEvaluator.from_config(
            config=retrieval_eval_config,
            llm_provider=llm_provider_mock,
        )
        query = qs_with_docs[0]
        doc = query.retrieved_docs[0]
        raw_answer, answer = evaluator.evaluate(query, doc)
        formatted_prompt = evaluator.prompt.format(query=query.query, document=doc.text)
        call_args = llm_provider_mock.call_mocker.call_args_list

        assert raw_answer == answer
        assert formatted_prompt in raw_answer
        assert call_args[0][0][0] == formatted_prompt


class TestCustomPromptEvaluator:
    def test_process_single_answer(
        self,
        llm_provider_json_mock,
        custom_prompt_retrieval_eval_config,
        qs_with_docs,
    ):
        query = qs_with_docs[0]
        doc = query.retrieved_docs[0]

        evaluator = CustomPromptEvaluator.from_config(
            config=custom_prompt_retrieval_eval_config,
            llm_provider=llm_provider_json_mock,
        )
        formatter = {
            custom_prompt_retrieval_eval_config.query_placeholder: query.query,
            custom_prompt_retrieval_eval_config.document_placeholder: doc.text,
        }
        formatted_prompt = custom_prompt_retrieval_eval_config.prompt.format(
            **formatter
        )

        _, answer = evaluator.evaluate(query, doc)
        call_args = llm_provider_json_mock.call_mocker.call_args_list

        assert answer == 0
        assert call_args[0][0][0] == formatted_prompt


class TestDomainExpertEvaluator:
    def test_sys_prompt(self, llm_provider_mock, expert_retrieval_eval_config):
        evaluator = DomainExpertEvaluator.from_config(
            config=expert_retrieval_eval_config, llm_provider=llm_provider_mock
        )
        assert expert_retrieval_eval_config.domain_long in evaluator.sys_prompt
        assert expert_retrieval_eval_config.domain_short in evaluator.sys_prompt
        assert (
            f"You work for {expert_retrieval_eval_config.company}"
            in evaluator.sys_prompt
        )

    def test_process_single_answer(
        self,
        llm_provider_mock_mock,
        expert_retrieval_eval_config,
        qs_with_docs,
    ):
        evaluator = DomainExpertEvaluator.from_config(
            config=expert_retrieval_eval_config,
            llm_provider=llm_provider_mock_mock,
        )
        results = evaluator.evaluate(qs_with_docs["0"]["0"])
        assert results["query_id"] == "0"
        assert results["did"] == "0"

        assert llm_provider_mock_mock.call_count == 2
        call_args = llm_provider_mock_mock.call_args_list
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
        assert prompts_score[1]["content"].endswith(
            expert_retrieval_eval_config.extra_guidelines
        )
        assert prompts_score[3]["content"].startswith("Given the previous reasoning")


class TestFewShotEvaluator:
    def test_process_single_answer(
        self,
        llm_provider_mock,
        few_shot_retrieval_eval_config,
        qs_with_docs,
    ):
        evaluator = FewShotEvaluator.from_config(
            config=few_shot_retrieval_eval_config, llm_provider=llm_provider_mock
        )
        results = evaluator.evaluate(qs_with_docs["0"]["0"])
        assert results["query_id"] == "0"
        assert results["did"] == "0"
        call_args = llm_provider_mock.call_mocker.call_args_list
        call_messages = call_args[0][0][0]
        assert len(call_messages) == 6
        assert call_messages[0]["role"] == "system"
        assert (
            call_messages[1]["role"]
            == call_messages[3]["role"]
            == call_messages[5]["role"]
            == "user"
        )
        assert call_messages[2]["role"] == call_messages[4]["role"] == "assistant"
