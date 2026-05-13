"""测试统计检验模块"""

import numpy as np
import pytest

from src.statistics import compare_two_groups, compare_multi_groups, significance_label, run_pipeline, stats_to_dataframe


class TestSignificanceLabel:
    def test_ns(self):
        assert significance_label(0.5) == "ns"
        assert significance_label(0.051) == "ns"

    def test_star(self):
        assert significance_label(0.04) == "*"

    def test_double_star(self):
        assert significance_label(0.005) == "**"

    def test_triple_star(self):
        assert significance_label(0.0005) == "***"

    def test_quad_star(self):
        # Three-level system: p<0.001 = *** (no four-star level)
        assert significance_label(0.00005) == "***"

    def test_nan(self):
        assert significance_label(np.nan) == ""


class TestCompareTwoGroups:
    def test_ttest_significant(self):
        a = np.array([1.0, 1.2, 0.9, 1.1, 1.0])
        b = np.array([4.0, 3.8, 4.1, 3.9, 4.2])
        result = compare_two_groups(a, b, test="ttest")
        assert result["p_value"] < 0.001
        assert result["significance"] in ("***", "****")

    def test_ttest_not_significant(self):
        a = np.array([1.0, 1.1, 0.9, 1.0, 1.1])
        b = np.array([1.0, 0.9, 1.1, 1.0, 1.0])
        result = compare_two_groups(a, b, test="ttest")
        assert result["p_value"] > 0.05
        assert result["significance"] == "ns"

    def test_mannwhitney(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
        result = compare_two_groups(a, b, test="mannwhitney")
        assert result["p_value"] < 0.05

    def test_small_sample(self):
        a = np.array([1.0])
        b = np.array([2.0])
        result = compare_two_groups(a, b)
        assert "error" in result


class TestMultiGroups:
    def test_three_groups_anova(self):
        groups = {
            "A": np.array([1.0, 1.1, 0.9, 1.0, 1.1]),
            "B": np.array([2.0, 2.1, 1.9, 2.0, 2.1]),
            "C": np.array([4.0, 3.9, 4.1, 4.0, 4.0]),
        }
        result = compare_multi_groups(groups)
        assert result.get("error") is None
        assert result["anova_result"]["p_value"] < 0.001

    def test_two_groups_falls_back_to_ttest(self):
        groups = {
            "A": np.array([1.0, 1.1, 0.9]),
            "B": np.array([2.0, 2.1, 1.9]),
        }
        result = compare_multi_groups(groups)
        assert "2 groups" in result.get("note", "")

    def test_with_control_comparison(self):
        groups = {
            "Control": np.array([1.0, 0.9, 1.1]),
            "TreatA": np.array([2.0, 2.1, 1.9]),
            "TreatB": np.array([3.0, 3.1, 2.9]),
        }
        result = compare_multi_groups(groups, control_label="Control")
        pairwise = result["pairwise_results"]
        assert "TreatA vs Control" in pairwise
        assert "TreatB vs Control" in pairwise


class TestRunPipeline:
    def test_two_groups_ttest(self):
        import pandas as pd
        result_df = pd.DataFrame({
            "sample_id": ["C1", "C2", "C3", "T1", "T2", "T3"],
            "group": ["Control"] * 3 + ["Treated"] * 3,
            "target_gene": ["IL6"] * 6,
            "delta_ct": [5.0, 4.9, 5.1, 3.0, 2.9, 3.1],  # stats on ΔCt now
            "fold_change": [1.0, 0.9, 1.1, 4.0, 4.1, 3.9],
        })
        stats = run_pipeline(result_df, "Control", test_method="ttest")
        assert "IL6" in stats
        pairwise = stats["IL6"]["pairwise_results"]
        comp = pairwise.get("Control vs Treated") or pairwise.get("Treated vs Control")
        assert comp is not None
