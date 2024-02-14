from ragelo.evaluators.retrieval_evaluators.base_retrieval_evaluator import (
    BaseRetrievalEvaluator,
)


class MockRetrievalEvaluator(BaseRetrievalEvaluator):
    def _build_message(self, qid: str, did: str) -> str:
        return f"Mock message for query {qid} and document {did}"

    def _process_answer(self, answer: str) -> str:
        return f"Processed {answer}"


class TestRetrievalEvaluator:
    def test_creation(self, llm_provider_mock, retrieval_eval_config):
        evaluator = MockRetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        assert len(evaluator) == 2

    def test_process_single_answer(self, llm_provider_mock, retrieval_eval_config):
        llm_provider_mock.return_value = "mocked API response"
        evaluator = MockRetrievalEvaluator.from_config(
            config=retrieval_eval_config, llm_provider=llm_provider_mock
        )
        results = evaluator.evaluate_single_sample("0", "0")
        assert results == "Processed mocked API response"
        call_args = llm_provider_mock.call_args_list
        assert call_args[0][0][0] == "Mock message for query 0 and document 0"

    # def test_process_single_answer(
    #     document_evaluator, mock_openai_client_call, mock_process_answer
    # ):
    #     # Set up the return values for the mock methods
    #     mock_openai_client_call.return_value = "mocked API response"
    #     mock_process_answer.return_value = "processed answer"

    #     # Call the method under test
    #     qid = "test_qid"
    #     did = "test_did"
    #     result = document_evaluator.evaluate_single_sample(qid, did)

    #     # Assert that the OpenAI client was called with the correct message
    #     expected_message = document_evaluator._build_message(qid, did)
    #     mock_openai_client_call.assert_called_once_with(
    #         document_evaluator.openai_client, expected_message
    #     )

    #     # Assert that the _process_answer method was called with the API response
    #     mock_process_answer.assert_called_once_with(
    #         document_evaluator, "mocked API response"
    #     )

    #     # Assert that the result is what we mocked _process_answer to return
    #     assert result == "processed answer"
