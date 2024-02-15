from ragelo.evaluators.answer_evaluators import BaseAnswerEvaluator
from ragelo.evaluators.answer_evaluators.pairwise_reasoning_evaluator import (
    PairwiseWithReasoningEvaluator,
)


class AnswerEvaluator(BaseAnswerEvaluator):
    def run(self):
        return


class TestAnswerEvaluator:
    def test_creation(self, llm_provider_mock, answer_eval_config):
        evaluator = AnswerEvaluator.from_config(
            config=answer_eval_config, llm_provider=llm_provider_mock
        )
        assert len(evaluator) == 2


class TestPairwiseWithReasoningEvaluator:
    def test_run(self, llm_provider_answer_mock, answer_eval_config):
        evaluator = PairwiseWithReasoningEvaluator.from_config(
            config=answer_eval_config, llm_provider=llm_provider_answer_mock
        )
        assert len(evaluator) == 2
        answers = evaluator.run()
        assert len(answers) == 4
        assert answers[0]["relevant"] == "A" and "[[A]]" in answers[0]["answer"]
        assert answers[1]["relevant"] == "B" and "[[B]]" in answers[1]["answer"]
        assert answers[2]["relevant"] == "C" and "[[C]]" in answers[2]["answer"]
        assert answers[3]["relevant"] == "C" and "[[C]]" in answers[3]["answer"]
        llm_call_args = llm_provider_answer_mock.inner_call.call_args_list
        assert len(llm_call_args) == 4
        assert isinstance(llm_call_args[0][0][0], str)
        assert llm_call_args[0][0][0] != llm_call_args[1][0][0]
