from ragelo import get_retrieval_evaluator
from ragelo.evaluators.retrieval_evaluators import (
    BaseRetrievalEvaluator,
    CustomPromptEvaluator,
    DomainExpertEvaluator,
    FewShotEvaluator,
    RDNAMEvaluator,
    ReasonerEvaluator,
)
from ragelo.types import Document


class RetrievalEvaluator(BaseRetrievalEvaluator):
    def _build_message(self, document: Document) -> str:
        if document.query is None:
            raise ValueError(f"Document {document.did} does not have a query.")
        qid = document.query.qid
        did = document.did
        return f"Mock message for query {qid} and document {did}"

    def _process_answer(self, answer: str) -> str:
        qid = answer.split("query")[1].strip().split(" ")[0]
        did = answer.split("document")[1].strip().split(" ")[0].split("\n")[0]
        return f"Processed answer for query {qid} and document {did}"


class TestRetrievalEvaluator:
    def test_process_single_answer(
        self, llm_provider_mock, retrieval_eval_config, documents_test
    ):
        evaluator = RetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        results = evaluator.evaluate_single_sample(documents_test["0"]["0"])
        assert results["query_id"] == "0"
        assert results["did"] == "0"
        assert (
            results["raw_answer"].split("\n")[0]
            == "Processed Mock message for query 0 and document 0"
        )
        call_args = llm_provider_mock.inner_call.call_args_list
        assert call_args[0][0][0] == "Mock message for query 0 and document 0"

    def test_run(self, llm_provider_mock, retrieval_eval_config, documents_test):
        evaluator = RetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        results = evaluator.run(documents_test)
        for qid in documents_test:
            for did in documents_test[qid]:
                assert (
                    results[qid][did]
                    == f"Processed answer for query {qid} and document {did}"
                )
        call_args = llm_provider_mock.inner_call.call_args_list
        assert call_args[0][0][0] == "Mock message for query 0 and document 0"
        assert call_args[1][0][0] == "Mock message for query 0 and document 1"
        assert call_args[2][0][0] == "Mock message for query 1 and document 2"
        assert call_args[3][0][0] == "Mock message for query 1 and document 3"

    def test_rich_printing(
        self, llm_provider_mock, retrieval_eval_config, documents_test, capsys
    ):
        retrieval_eval_config.rich_print = True
        evaluator = RetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        _ = evaluator.run(documents_test)
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
        self, llm_provider_mock_rdnam, rdnam_config, documents_test
    ):
        evaluator = RDNAMEvaluator.from_config(
            config=rdnam_config, llm_provider=llm_provider_mock_rdnam
        )
        results = evaluator.evaluate_single_sample(documents_test["0"]["0"])
        assert isinstance(results, dict)
        assert results["answer"] == 1
        assert results["query_id"] == "0"
        assert results["did"] == "0"
        call_args = llm_provider_mock_rdnam.inner_call.call_args_list
        assert call_args[0][0][0].startswith(
            "You are a search quality rater evaluating"
        )


class TestReasonerEvaluator:
    def test_process_single_answer(
        self, llm_provider_mock, retrieval_eval_config, documents_test
    ):
        evaluator = ReasonerEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        results = evaluator.evaluate_single_sample(documents_test["0"]["0"])
        assert results["query_id"] == "0"
        assert results["did"] == "0"

        formatted_prompt = evaluator.prompt.format(
            query=documents_test["0"]["0"].query.query,
            document=documents_test["0"]["0"].text,
        )
        assert formatted_prompt in results["raw_answer"]
        assert results["raw_answer"] == results["answer"]
        call_args = llm_provider_mock.inner_call.call_args_list
        assert call_args[0][0][0] == formatted_prompt


class TestCustomPromptEvaluator:
    def test_process_single_answer(
        self, llm_provider_mock, custom_prompt_retrieval_eval_config, documents_test
    ):
        evaluator = CustomPromptEvaluator.from_config(
            config=custom_prompt_retrieval_eval_config, llm_provider=llm_provider_mock
        )
        formatter = {
            custom_prompt_retrieval_eval_config.query_placeholder: documents_test["0"][
                "0"
            ].query.query,
            custom_prompt_retrieval_eval_config.document_placeholder: documents_test[
                "0"
            ]["0"].text,
        }
        formatted_prompt = custom_prompt_retrieval_eval_config.prompt.format(
            **formatter
        )
        results = evaluator.evaluate_single_sample(documents_test["0"]["0"])
        assert results["query_id"] == "0"
        assert results["did"] == "0"

        call_args = llm_provider_mock.inner_call.call_args_list
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
        self, llm_provider_mock_mock, expert_retrieval_eval_config, documents_test
    ):
        evaluator = DomainExpertEvaluator.from_config(
            config=expert_retrieval_eval_config,
            llm_provider=llm_provider_mock_mock,
        )
        results = evaluator.evaluate_single_sample(documents_test["0"]["0"])
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
        self, llm_provider_mock, few_shot_retrieval_eval_config, documents_test
    ):
        evaluator = FewShotEvaluator.from_config(
            config=few_shot_retrieval_eval_config, llm_provider=llm_provider_mock
        )
        results = evaluator.evaluate_single_sample(documents_test["0"]["0"])
        assert results["query_id"] == "0"
        assert results["did"] == "0"
        call_args = llm_provider_mock.inner_call.call_args_list
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
