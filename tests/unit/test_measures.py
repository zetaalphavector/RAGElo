import pytest

ir_measures = pytest.importorskip("ir_measures")

from ragelo.measures import make_run, paired_permutation_pvalue, parse_measure  # noqa: E402


class TestParseMeasure:
    @pytest.mark.parametrize(
        "metric",
        [
            "nDCG@10",
            "R@50",
            "Judged@10",
            "AP",
            "AP(rel=2)@100",
            "alpha_nDCG@20",
            "StRecall@50",
        ],
    )
    def test_resolves_measure_strings_without_the_ast_based_parser(self, metric):
        """`ir_measures.parse_measure` evaluates parameters through `ast.Num`, removed in Python 3.14."""
        assert str(parse_measure(metric)) == metric

    def test_measure_objects_pass_through(self):
        measure = ir_measures.nDCG @ 10
        assert parse_measure(measure) is measure

    @pytest.mark.parametrize("metric", ["NotAMeasure@10", "nDCG(rel)@10"])
    def test_unresolvable_metric_raises_value_error(self, metric):
        with pytest.raises(ValueError):
            parse_measure(metric)


class TestPairedPermutationPvalue:
    def test_no_observed_difference_is_never_significant(self):
        assert paired_permutation_pvalue([]) == 1.0
        assert paired_permutation_pvalue([0.0] * 10) == 1.0

    def test_consistent_gains_are_significant(self):
        assert paired_permutation_pvalue([0.1] * 20) < 0.05

    def test_noise_is_not_significant(self):
        assert paired_permutation_pvalue([0.3, -0.28, 0.05, -0.07, 0.01, -0.02]) > 0.05


class TestQrelConstruction:
    def test_make_run_flattens_the_nested_run_mapping(self):
        run = make_run({"q1": {"d1": 2.0, "d2": 1.0}, "q2": {"d3": 3.0}})
        assert {(d.query_id, d.doc_id, d.score) for d in run} == {
            ("q1", "d1", 2.0),
            ("q1", "d2", 1.0),
            ("q2", "d3", 3.0),
        }
