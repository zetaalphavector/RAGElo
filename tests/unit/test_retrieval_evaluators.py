from unittest.mock import patch

from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
)


class BaseRetrievalEvaluator(BaseRetrievalEvaluator):
    def _build_message(self, qid: str, did: str) -> str:
        return f"Mock message for query {qid} and document {did}"

    def _process_answer(self, answer: str) -> str:
        qid = answer.split("query")[1].strip().split(" ")[0]
        did = answer.split("document")[1].strip().split(" ")[0]
        return f"Processed answer for query {qid} and document {did}"


class TestRetrievalEvaluator:
    def test_creation(self, llm_provider, retrieval_eval_config):
        evaluator = BaseRetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider
        )
        assert len(evaluator) == 2

    def test_process_single_answer(self, llm_provider_mock, retrieval_eval_config):
        # mock_llm_provider.return_value = "mocked API response"
        evaluator = BaseRetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        results = evaluator.evaluate_single_sample("0", "0")
        assert results == "Processed answer for query 0 and document 0"
        call_args = llm_provider_mock.inner_call.call_args_list
        assert call_args[0][0][0] == "Mock message for query 0 and document 0"

    def test_run(self, llm_provider_mock, retrieval_eval_config):
        evaluator = BaseRetrievalEvaluator.from_config(
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
