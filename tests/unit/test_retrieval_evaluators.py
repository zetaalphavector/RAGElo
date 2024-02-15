from ragelo.evaluators.retrieval_evaluators import (
    BaseRetrievalEvaluator,
    DomainExpertEvaluator,
    RDNAMEvaluator,
    ReasonerEvaluator,
)


class RetrievalEvaluator(BaseRetrievalEvaluator):
    def _build_message(self, qid: str, did: str) -> str:
        return f"Mock message for query {qid} and document {did}"

    def _process_answer(self, answer: str) -> str:
        qid = answer.split("query")[1].strip().split(" ")[0]
        did = answer.split("document")[1].strip().split(" ")[0]
        return f"Processed answer for query {qid} and document {did}"


class TestRetrievalEvaluator:
    def test_creation(self, llm_provider_mock, retrieval_eval_config):
        evaluator = RetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        assert len(evaluator) == 2

    def test_process_single_answer(self, llm_provider_mock, retrieval_eval_config):
        evaluator = RetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        results = evaluator.evaluate_single_sample("0", "0")
        assert results == "Processed answer for query 0 and document 0"
        call_args = llm_provider_mock.inner_call.call_args_list
        assert call_args[0][0][0] == "Mock message for query 0 and document 0"

    def test_run(self, llm_provider_mock, retrieval_eval_config):
        evaluator = RetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        results = evaluator.run()
        assert results == {
            "0": {
                "0": "Processed answer for query 0 and document 0",
                "1": "Processed answer for query 0 and document 1",
            },
            "1": {
                "2": "Processed answer for query 1 and document 2",
                "3": "Processed answer for query 1 and document 3",
            },
        }
        call_args = llm_provider_mock.inner_call.call_args_list
        assert call_args[0][0][0] == "Mock message for query 0 and document 0"
        assert call_args[1][0][0] == "Mock message for query 0 and document 1"
        assert call_args[2][0][0] == "Mock message for query 1 and document 2"
        assert call_args[3][0][0] == "Mock message for query 1 and document 3"

    def test_rich_printing(self, llm_provider_mock, retrieval_eval_config, capsys):
        retrieval_eval_config.rich_print = True
        evaluator = RetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        _ = evaluator.run()
        captured = capsys.readouterr()
        assert "🔎" in captured.out


class TestRDNAMEvaluator:
    def test_creation(self, llm_provider_mock, rdnam_config):
        evaluator = RDNAMEvaluator.from_config(
            config=rdnam_config, llm_provider=llm_provider_mock
        )
        assert len(evaluator) == 2

    def test_process_single_answer(self, llm_provider_mock_rdnam, rdnam_config):
        evaluator = RDNAMEvaluator.from_config(
            config=rdnam_config, llm_provider=llm_provider_mock_rdnam
        )
        results = evaluator.evaluate_single_sample("0", "0")
        assert results == 1
        call_args = llm_provider_mock_rdnam.inner_call.call_args_list
        assert call_args[0][0][0].startswith(
            "You are a search quality rater evaluating"
        )


class TestReasonerEvaluator:
    def test_creation(self, llm_provider_mock, retrieval_eval_config):
        evaluator = ReasonerEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        assert len(evaluator) == 2

    def test_process_single_answer(self, llm_provider_mock, retrieval_eval_config):
        evaluator = ReasonerEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        results = evaluator.evaluate_single_sample("0", "0")
        formatted_prompt = evaluator.prompt.format(
            user_question=evaluator.queries["0"].query,
            doc_content=evaluator.documents["0"]["0"].text,
        )
        assert formatted_prompt in results
        call_args = llm_provider_mock.inner_call.call_args_list
        assert call_args[0][0][0] == formatted_prompt


class TestDomainExpertEvaluator:
    def test_sys_prompt(self, llm_provider_mock, expert_retrieval_eval_config):
        evaluator = DomainExpertEvaluator.from_config(
            config=expert_retrieval_eval_config, llm_provider=llm_provider_mock
        )
        assert len(evaluator) == 2
        assert expert_retrieval_eval_config.domain_long in evaluator.sys_prompt
        assert expert_retrieval_eval_config.domain_short in evaluator.sys_prompt
        assert (
            f"You work for {expert_retrieval_eval_config.company}"
            in evaluator.sys_prompt
        )

    def test_process_single_answer(
        self, llm_provider_mock_mock, expert_retrieval_eval_config
    ):
        evaluator = DomainExpertEvaluator.from_config(
            config=expert_retrieval_eval_config,
            llm_provider=llm_provider_mock_mock,
        )
        _ = evaluator.evaluate_single_sample("0", "0")

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
