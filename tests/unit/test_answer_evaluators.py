from ragelo import get_answer_evaluator
from ragelo.evaluators.answer_evaluators.pairwise_reasoning_evaluator import (
    PairwiseWithReasoningEvaluator,
)


class TestPairwiseWithReasoningEvaluator:
    def test_run(
        self,
        llm_provider_answer_mock,
        pairwise_answer_eval_config,
        answers_test,
    ):
        evaluator = PairwiseWithReasoningEvaluator.from_config(
            config=pairwise_answer_eval_config, llm_provider=llm_provider_answer_mock
        )
        answers = evaluator.run(answers_test)
        assert len(answers) == 2
        assert answers[0]["answer"] == "A" and "[[A]]" in answers[0]["raw_answer"]
        assert answers[1]["answer"] == "B" and "[[B]]" in answers[1]["raw_answer"]
        llm_call_args = llm_provider_answer_mock.inner_call.call_args_list
        assert len(llm_call_args) == 2
        assert isinstance(llm_call_args[0][0][0], str)
        assert llm_call_args[0][0][0] != llm_call_args[1][0][0]
        # Make sure that no games with the same agent were called
        for call_arg in llm_call_args:
            agent_a_answer = (
                call_arg[0][0]
                .split("[The Start of Assistant A's Answer]")[1]
                .split("[The End of Assistant A's Answer]")[0]
            ).strip()
            agent_b_answer = (
                call_arg[0][0]
                .split("[The Start of Assistant B's Answer]")[1]
                .split("[The End of Assistant B's Answer]")[0]
            ).strip()
            assert agent_a_answer != agent_b_answer


def test_get_by_name(llm_provider_answer_mock, pairwise_answer_eval_config):
    evaluator = get_answer_evaluator(
        "pairwise_reasoning",
        llm_provider_answer_mock,
        reasoning_path=pairwise_answer_eval_config.reasoning_path,
    )
    assert isinstance(evaluator, PairwiseWithReasoningEvaluator)
