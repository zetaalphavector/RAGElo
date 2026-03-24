import pytest

from ragelo.evaluators.retrieval_evaluators import BaseRetrievalEvaluator
from ragelo.types import Document, Query
from ragelo.types.answer_formats import RetrievalEvaluationAnswer
from ragelo.types.formats import LLMInputPrompt
from ragelo.types.results import RetrievalEvaluatorResult


class _TestRetrievalEvaluator(BaseRetrievalEvaluator):
    def _build_message(self, query: Query, document: Document) -> LLMInputPrompt:
        return LLMInputPrompt(user_message=f"Query: {query.query}\nDocument: {document.text}")


class TestEvaluateExperimentAsync:
    @pytest.mark.asyncio
    async def test_yields_results(self, llm_provider_mock_retrieval, experiment, base_retrieval_eval_config):
        evaluator = _TestRetrievalEvaluator.from_config(
            config=base_retrieval_eval_config, llm_provider=llm_provider_mock_retrieval
        )
        results = []
        async for eval_tuple, evaluation in evaluator.evaluate_experiment_async(experiment):
            results.append((eval_tuple, evaluation))

        # 2 queries × 2 docs = 4 evaluations
        assert len(results) == 4
        for (query, doc), evaluation in results:
            assert isinstance(query, Query)
            assert isinstance(doc, Document)
            assert isinstance(evaluation, RetrievalEvaluatorResult)
            assert isinstance(evaluation.answer, RetrievalEvaluationAnswer)

    @pytest.mark.asyncio
    async def test_on_result_callback(self, llm_provider_mock_retrieval, experiment, base_retrieval_eval_config):
        evaluator = _TestRetrievalEvaluator.from_config(
            config=base_retrieval_eval_config, llm_provider=llm_provider_mock_retrieval
        )
        callback_results = []

        def on_result(eval_tuple, evaluation):
            callback_results.append((eval_tuple, evaluation))

        yielded = []
        async for item in evaluator.evaluate_experiment_async(experiment, on_result=on_result):
            yielded.append(item)

        assert len(callback_results) == 4
        assert len(yielded) == 4
        # Callback and yield should produce the same items
        for cb_item, yield_item in zip(callback_results, yielded):
            assert cb_item == yield_item

    def test_sync_wrapper_still_works(self, llm_provider_mock_retrieval, experiment, base_retrieval_eval_config):
        evaluator = _TestRetrievalEvaluator.from_config(
            config=base_retrieval_eval_config, llm_provider=llm_provider_mock_retrieval
        )
        evaluator.evaluate_experiment(experiment)

        evaluator_name = str(base_retrieval_eval_config.evaluator_name)
        for query in experiment:
            for doc in query.retrieved_docs_iter():
                assert evaluator_name in doc.evaluations
                assert isinstance(doc.evaluations[evaluator_name], RetrievalEvaluatorResult)
