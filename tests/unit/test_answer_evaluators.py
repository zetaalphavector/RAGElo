from typing import cast

from ragelo import get_answer_evaluator
from ragelo.evaluators.answer_evaluators import (
    CustomPromptEvaluator,
    PairwiseWithReasoningEvaluator,
)


class TestPairwiseWithReasoningEvaluator:
    def test_run(
        self,
        llm_provider_pairwise_answer_mock,
        pairwise_answer_eval_config,
        answers_test,
    ):
        evaluator = PairwiseWithReasoningEvaluator.from_config(
            config=pairwise_answer_eval_config,
            llm_provider=llm_provider_pairwise_answer_mock,
        )
        answers = evaluator.batch_evaluate(answers_test)
        assert len(answers) == 4
        assert answers[0].answer == "A" and "[[A]]" in answers[0].raw_answer
        assert answers[1].answer == "B" and "[[B]]" in answers[1].raw_answer
        assert answers[2].answer == "C" and "[[C]]" in answers[2].raw_answer
        assert answers[3].answer == "C" and "[[C]]" in answers[2].raw_answer

        llm_call_args = llm_provider_pairwise_answer_mock.call_mocker.call_args_list
        assert len(llm_call_args) == 4
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


class TestCustomPromptEvaluator:
    def test_run(
        self, llm_provider_answer_mock, custom_answer_eval_config, answers_test
    ):
        evaluator = CustomPromptEvaluator.from_config(
            config=custom_answer_eval_config,
            llm_provider=llm_provider_answer_mock,
        )
        answers = evaluator.batch_evaluate(answers_test)
        flat_answers = [(q, a) for q in answers_test for a in q.answers]
        assert all([isinstance(a.answer, dict) for a in answers])
        assert len(answers) == 4
        assert answers[0].agent == answers[2].agent == "agent1"
        assert answers[1].agent == answers[3].agent == "agent2"

        parsed_answer = cast(dict[str, int], answers[0].answer)
        assert parsed_answer["quality"] == 2
        assert parsed_answer["trustworthiness"] == 1
        assert parsed_answer["originality"] == 1

        llm_call_args = llm_provider_answer_mock.call_mocker.call_args_list
        assert (
            len(
                llm_call_args[0][0][0]
                .split("DOCUMENTS RETRIEVED:")[1]
                .split("User Query")[0]
                .strip()
                .split("\n")
            )
            == 2
        )

        flat_answers = [(q, a) for q in answers_test for a in q.answers]
        for (q, a), args in zip(flat_answers, llm_call_args):
            submitted_query = args[0][0].split("User Query: ")[1].split("\n")[0].strip()
            submitted_answer = (
                args[0][0].split("Agent answer: ")[1].split("\n")[-1].strip()
            )
            expected_query = q.query
            expected_answer = a.text
            assert submitted_query == expected_query
            assert submitted_answer == expected_answer


def test_get_by_name(llm_provider_pairwise_answer_mock, pairwise_answer_eval_config):
    evaluator = get_answer_evaluator(
        "pairwise_reasoning",
        llm_provider_pairwise_answer_mock,
        reasoning_path=pairwise_answer_eval_config.reasoning_path,
    )
    assert isinstance(evaluator, PairwiseWithReasoningEvaluator)
