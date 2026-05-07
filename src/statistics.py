# -*- coding: utf-8 -*-
"""Statistical tests: t-test / ANOVA / post-hoc.

All tests use DeltaCt (log-scale) values — NOT fold_change or normalized_data.
This matches GraphPad Prism's approach of testing on the log-transformed data.
"""

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_TUKEY = True
except ImportError:
    HAS_TUKEY = False


def significance_label(p: float) -> str:
    """Three-level: p<0.001 (***), p<0.01 (**), p<0.05 (*), else ns."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def is_significant(p: float) -> bool:
    return p is not None and not np.isnan(p) and p < 0.05


def compare_two_groups(
    group_a: np.ndarray,
    group_b: np.ndarray,
    test: str = "ttest",
    alternative: str = "two-sided",
) -> dict:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a_clean = a[~np.isnan(a)]
    b_clean = b[~np.isnan(b)]

    if len(a_clean) < 2 or len(b_clean) < 2:
        return {
            "statistic": np.nan, "p_value": np.nan,
            "significance": "", "test_name": test,
            "error": "Insufficient sample size",
        }

    if test == "mannwhitney":
        stat, p = stats.mannwhitneyu(a_clean, b_clean, alternative=alternative)
        test_name = "Mann-Whitney U"
    else:
        stat, p = stats.ttest_ind(a_clean, b_clean, alternative=alternative)
        test_name = "Student's t-test"

    return {
        "statistic": stat,
        "p_value": p,
        "significance": significance_label(p),
        "test_name": test_name,
        "n_a": len(a_clean),
        "n_b": len(b_clean),
        "mean_a": np.mean(a_clean),
        "mean_b": np.mean(b_clean),
    }


def compare_multi_groups(
    groups: dict[str, np.ndarray],
    control_label: str | None = None,
) -> dict:
    group_data = {k: np.asarray(v, dtype=float) for k, v in groups.items()}
    clean_data = {
        k: v[~np.isnan(v)]
        for k, v in group_data.items()
        if len(v[~np.isnan(v)]) >= 2
    }

    if len(clean_data) < 2:
        return {"error": "Valid groups < 2"}

    if len(clean_data) == 2:
        keys = list(clean_data.keys())
        pairwise = {
            f"{keys[0]} vs {keys[1]}": compare_two_groups(
                clean_data[keys[0]], clean_data[keys[1]]
            )
        }
        return {
            "anova_result": None,
            "pairwise_results": pairwise,
            "note": "2 groups, using t-test",
        }

    # 3+ groups: ANOVA
    arrays = list(clean_data.values())
    f_stat, p_anova = stats.f_oneway(*arrays)
    anova = {
        "statistic": f_stat,
        "p_value": p_anova,
        "significance": significance_label(p_anova),
        "test_name": "One-way ANOVA",
        "n_groups": len(clean_data),
    }

    # Pairwise vs control
    pairwise = {}
    if control_label and control_label in clean_data:
        ctrl_vals = clean_data[control_label]
        for label, vals in clean_data.items():
            if label == control_label:
                continue
            pairwise[f"{label} vs {control_label}"] = compare_two_groups(
                vals, ctrl_vals
            )

    return {"anova_result": anova, "pairwise_results": pairwise}


def run_pipeline(
    result_df: pd.DataFrame,
    control_group: str,
    test_method: str = "auto",
) -> dict:
    """Run statistics on DeltaCt values per target gene.

    Returns dict: {target_gene: {anova_result, pairwise_results}}
    """
    all_results = {}
    n_groups = result_df["group"].nunique()

    for target in result_df["target_gene"].unique():
        tg_data = result_df[result_df["target_gene"] == target]
        # Use DeltaCt (log-scale) — NOT fold_change / normalized_data
        groups = tg_data.groupby("group")["delta_ct"].apply(list).to_dict()

        group_keys = list(groups.keys())
        if len(group_keys) < 2:
            all_results[target] = {"error": "Groups < 2"}
            continue

        if len(group_keys) == 2 or n_groups == 2:
            use_method = "ttest"
            if test_method == "mannwhitney":
                use_method = "mannwhitney"

            raw = compare_two_groups(
                np.array(groups[group_keys[0]]),
                np.array(groups[group_keys[1]]),
                test=use_method,
            )
            all_results[target] = {
                "pairwise_results": {
                    f"{group_keys[0]} vs {group_keys[1]}": raw,
                }
            }
        else:
            clean_groups = {
                k: np.array(v)
                for k, v in groups.items()
                if len([x for x in v if pd.notna(x)]) >= 2
            }
            anova_result = None
            if len(clean_groups) >= 3:
                arrays = list(clean_groups.values())
                f_stat, p_anova = stats.f_oneway(*arrays)
                anova_result = {
                    "statistic": f_stat,
                    "p_value": p_anova,
                    "significance": significance_label(p_anova),
                    "test_name": "One-way ANOVA",
                    "n_groups": len(clean_groups),
                }

            pairwise = {}
            if control_group in clean_groups:
                ctrl_vals = clean_groups[control_group]
                for label, vals in clean_groups.items():
                    if label == control_group:
                        continue
                    pairwise[f"{label} vs {control_group}"] = compare_two_groups(
                        vals, ctrl_vals
                    )

            all_results[target] = {
                "anova_result": anova_result,
                "pairwise_results": pairwise,
            }

    return all_results


def stats_to_dataframe(stats_results: dict) -> pd.DataFrame:
    """Convert stats results into a display-ready DataFrame.

    Columns: 目的基因 | 比较组别 | 检验方法 | 统计量 | P值 | 显著性
    """
    rows = []
    for gene, sres in stats_results.items():
        anova = sres.get("anova_result")
        pairwise = sres.get("pairwise_results", {})

        if anova and anova.get("p_value") is not None:
            rows.append({
                "目的基因": gene,
                "比较组别": f"所有组 (n={anova.get('n_groups', '?')})",
                "检验方法": anova.get("test_name", ""),
                "统计量": f"F = {anova['statistic']:.4f}",
                "P值": f"{anova['p_value']:.6f}",
                "显著性": anova.get("significance", ""),
            })

        for comp, detail in pairwise.items():
            if isinstance(detail, dict):
                rows.append({
                    "目的基因": gene,
                    "比较组别": comp,
                    "检验方法": detail.get("test_name", ""),
                    "统计量": f"t = {detail.get('statistic', 0):.4f}",
                    "P值": f"{detail.get('p_value', 1):.6f}",
                    "显著性": detail.get("significance", ""),
                })

    if not rows:
        return pd.DataFrame({"提示": ["无统计检验结果"]})
    return pd.DataFrame(rows)


def get_significance_brackets(
    stats_results: dict,
    groups: list[str],
    target_gene: str,
    group_to_x: dict[str, int] | None = None,
) -> list[dict]:
    """Extract pairwise sig comparisons and return bracket coordinates.

    Returns list of dicts with: x0, x1, y_top, label, p_value
    """
    brackets = []
    if target_gene not in stats_results:
        return brackets

    pairwise = stats_results[target_gene].get("pairwise_results", {})

    for comp, detail in pairwise.items():
        if not isinstance(detail, dict):
            continue
        if not is_significant(detail.get("p_value", np.nan)):
            continue

        # Parse "A vs B" or "GroupA vs GroupB"
        parts = comp.split(" vs ")
        if len(parts) != 2:
            continue
        g_a, g_b = parts[0].strip(), parts[1].strip()

        if group_to_x:
            x0 = group_to_x.get(g_a)
            x1 = group_to_x.get(g_b)
            if x0 is None or x1 is None:
                continue
        else:
            try:
                x0 = groups.index(g_a)
                x1 = groups.index(g_b)
            except ValueError:
                continue

        brackets.append({
            "x0": min(x0, x1),
            "x1": max(x0, x1),
            "label": detail.get("significance", "*"),
            "p_value": detail.get("p_value", np.nan),
        })

    return brackets
