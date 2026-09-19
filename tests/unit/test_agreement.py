import math

import pytest

from benchmarks.agreement import agreement


class TestAgreement:
    def test_matches_sklearn_krippendorff_and_scipy_over_the_pairs_both_sides_judged(self):
        """Expected values come from sklearn's cohen_kappa_score, krippendorff's alpha and scipy's spearmanr.

        p7 has no LLM label and p9 no human one, so neither counts. The human 3 collapses to a 2 for
        kappa and alpha, which need both sides on one scale.
        """
        human = {"q1": {"p1": 3, "p2": 2, "p3": 1, "p4": 0}, "q2": {"p5": 0, "p6": 3, "p7": 1}}
        judged = {"q1": {"p1": 2, "p2": 2, "p3": 1, "p4": 0, "p9": 2}, "q2": {"p5": 1, "p6": 1}}

        result = agreement(human, judged)

        assert result.n_pairs == 6
        assert result.kappa_binary == pytest.approx(0.6666667)
        assert result.kappa_graded == pytest.approx(0.52)
        assert result.alpha_ordinal == pytest.approx(0.7635582)
        assert result.spearman == pytest.approx(0.6515328)

    def test_a_judge_that_gives_one_label_to_everything_has_no_defined_agreement(self):
        human = {"q1": {"p1": 3, "p2": 0}}
        judged = {"q1": {"p1": 2, "p2": 2}}

        result = agreement(human, judged)

        assert result.kappa_graded == 0
        assert math.isnan(result.spearman)
