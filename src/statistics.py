# -*- coding: utf-8 -*-
"""Statistical tests: t-test / ANOVA / post-hoc with auto normality detection.

All tests operate on user-selectable data column (default: normalized_data / Fold Change).
Auto-detects normality (Shapiro-Wilk) and variance homogeneity (Levene) to choose between
parametric (ANOVA + Tukey HSD) and non-parametric (Kruskal-Wallis + Dunn) paths.
Outputs full pairwise P-value matrix — all groups vs all groups.
"""

import itertools
import numpy as np
import pandas as pd
from scipy import stats

try:
    from scipy.stats import tukey_hsd as _scipy_tukey
    HAS_SCIPY_TUKEY = True
except ImportError:
    HAS_SCIPY_TUKEY = False

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    HAS_STATSMODELS_TUKEY = True
except ImportError:
    HAS_STATSMODELS_TUKEY = False


# ── significance helpers ──────────────────────────────────────

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


# ── distribution diagnostics ──────────────────────────────────

def test_normality(values: np.ndarray) -> dict:
    """Shapiro-Wilk normality test."""
    clean = np.asarray(values, dtype=float)
    clean = clean[~np.isnan(clean)]
    if len(clean) < 3:
        return {"statistic": np.nan, "p_value": np.nan, "is_normal": False,
                "reason": "n < 3"}
    stat, p = stats.shapiro(clean)
    return {"statistic": stat, "p_value": p, "is_normal": p >= 0.05,
            "test": "Shapiro-Wilk"}


def test_equal_variance(groups: dict[str, np.ndarray]) -> dict:
    """Levene's test for homogeneity of variances."""
    arrays = [np.asarray(v, dtype=float) for v in groups.values()]
    arrays = [a[~np.isnan(a)] for a in arrays if len(a[~np.isnan(a)]) >= 2]
    if len(arrays) < 2:
        return {"statistic": np.nan, "p_value": np.nan, "equal_var": False,
                "reason": "valid groups < 2"}
    stat, p = stats.levene(*arrays)
    return {"statistic": stat, "p_value": p, "equal_var": p >= 0.05,
            "test": "Levene"}


# ── two-group comparison ──────────────────────────────────────

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


# ── full pairwise post-hoc ────────────────────────────────────

def _tukey_pairwise(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Tukey's HSD: return full pairwise P-value matrix."""
    labels = list(groups.keys())
    n = len(labels)
    mat = pd.DataFrame(np.eye(n), index=labels, columns=labels)

    if HAS_SCIPY_TUKEY:
        arrays = [np.asarray(groups[g], dtype=float) for g in labels]
        result = _scipy_tukey(*arrays)
        pvals = result.pvalue
        for i, j in itertools.combinations(range(n), 2):
            mat.iloc[i, j] = pvals[i][j] if i < j else pvals[j][i]
            mat.iloc[j, i] = mat.iloc[i, j]
    elif HAS_STATSMODELS_TUKEY:
        all_vals = np.concatenate([np.asarray(groups[g], dtype=float) for g in labels])
        all_labels = np.concatenate([[g] * len(groups[g]) for g in labels])
        tukey = pairwise_tukeyhsd(all_vals, all_labels, alpha=0.05)
        for i, j in itertools.combinations(range(n), 2):
            # statsmodels stores results as (group1, group2, meandiff, p-adj, lower, upper, reject)
            for row in tukey.summary().data[1:]:
                g1, g2 = row[0].strip(), row[1].strip()
                p = float(row[3])
                if (g1 == labels[i] and g2 == labels[j]) or (g1 == labels[j] and g2 == labels[i]):
                    mat.iloc[i, j] = p
                    mat.iloc[j, i] = p
                    break
    else:
        # Fallback: manual Tukey using studentized range
        for i, j in itertools.combinations(range(n), 2):
            a = np.asarray(groups[labels[i]], dtype=float)
            b = np.asarray(groups[labels[j]], dtype=float)
            a, b = a[~np.isnan(a)], b[~np.isnan(b)]
            if len(a) < 2 or len(b) < 2:
                mat.iloc[i, j] = np.nan
                mat.iloc[j, i] = np.nan
                continue
            se = np.sqrt(0.5 * (np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b)))
            if se == 0:
                mat.iloc[i, j] = 1.0
            else:
                q = abs(np.mean(a) - np.mean(b)) / se
                # Approximate: use t-distribution with Bonferroni
                df = len(a) + len(b) - 2
                p_raw = 2 * stats.t.sf(q / np.sqrt(2), df)
                mat.iloc[i, j] = min(p_raw * (n * (n - 1) / 2), 1.0)
            mat.iloc[j, i] = mat.iloc[i, j]
    return mat


def _dunn_pairwise(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Dunn's test: full pairwise P-value matrix with Bonferroni correction."""
    labels = list(groups.keys())
    n = len(labels)

    # Collect all values with group labels
    all_vals = []
    all_grps = []
    for g in labels:
        arr = np.asarray(groups[g], dtype=float)
        arr_clean = arr[~np.isnan(arr)]
        all_vals.extend(arr_clean.tolist())
        all_grps.extend([g] * len(arr_clean))
    all_vals = np.array(all_vals)

    # Rank all values (average rank for ties)
    from scipy.stats import rankdata
    ranks = rankdata(all_vals)
    N = len(ranks)

    # Mean rank per group
    rank_sums = {}
    n_per_group = {}
    for g in labels:
        mask = np.array(all_grps) == g
        rank_sums[g] = ranks[mask].sum()
        n_per_group[g] = mask.sum()

    # Pairwise z-scores
    mat = pd.DataFrame(np.eye(n), index=labels, columns=labels)
    n_comparisons = n * (n - 1) / 2

    for i, j in itertools.combinations(range(n), 2):
        gi, gj = labels[i], labels[j]
        ni, nj = n_per_group[gi], n_per_group[gj]
        if ni < 2 or nj < 2:
            mat.iloc[i, j] = np.nan
            mat.iloc[j, i] = np.nan
            continue
        r_bar_i = rank_sums[gi] / ni
        r_bar_j = rank_sums[gj] / nj
        denom = np.sqrt((N * (N + 1) / 12.0) * (1.0 / ni + 1.0 / nj))
        if denom == 0:
            mat.iloc[i, j] = 1.0
        else:
            z = abs(r_bar_i - r_bar_j) / denom
            p_raw = 2 * stats.norm.sf(z)
            p_adj = min(p_raw * n_comparisons, 1.0)
            mat.iloc[i, j] = p_adj
        mat.iloc[j, i] = mat.iloc[i, j]
    return mat


# ── main pipeline ─────────────────────────────────────────────

def run_pipeline(
    result_df: pd.DataFrame,
    control_group: str,
    test_method: str = "auto",
    data_col: str = "normalized_data",
) -> dict:
    """Run full statistics pipeline per target gene.

    Args:
        result_df: From calculator.compute().
        control_group: Label of the control/reference group.
        test_method: "auto" (default), "ttest", "mannwhitney", "anova",
                     "kruskal", or "none".
        data_col: Column to test — "normalized_data" (fold change) or
                  "delta_ct" (log-scale, GraphPad Prism convention).

    Returns:
        {target_gene: {
            normality: {group: dict, ...},
            variance_homogeneity: dict,
            test_path: "parametric" | "nonparametric" | "twogroup",
            omnibus_result: dict (ANOVA or Kruskal-Wallis),
            pairwise_matrix: DataFrame (all-vs-all P-values),
            pairwise_sig_brackets: [dict, ...],
        }}
    """
    all_results = {}
    n_groups = result_df["group"].nunique()

    for target in result_df["target_gene"].unique():
        tg_data = result_df[result_df["target_gene"] == target].copy()

        if data_col not in tg_data.columns:
            all_results[target] = {"error": f"Column '{data_col}' not found"}
            continue

        groups_raw = tg_data.groupby("group")[data_col].apply(list).to_dict()
        group_keys = list(groups_raw.keys())

        if len(group_keys) < 2:
            all_results[target] = {"error": "Groups < 2"}
            continue

        # ── Clean data ──
        groups_clean = {}
        for k, v in groups_raw.items():
            arr = np.asarray(v, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) >= 2:
                groups_clean[k] = arr

        if len(groups_clean) < 2:
            all_results[target] = {"error": "Valid groups (n≥2) < 2"}
            continue

        # ── Normality + variance diagnostics ──
        normality = {g: test_normality(groups_clean[g]) for g in groups_clean}
        var_result = test_equal_variance(groups_clean)

        all_normal = all(n["is_normal"] for n in normality.values())
        equal_var = var_result.get("equal_var", False)

        # ── Determine test path ──
        if test_method == "none":
            all_results[target] = {"error": "Skipped", "normality": normality}
            continue

        if len(groups_clean) == 2:
            # Two-group path
            keys = list(groups_clean.keys())
            if test_method == "mannwhitney":
                use_test = "mannwhitney"
            elif test_method == "ttest":
                use_test = "ttest"
            elif test_method == "auto":
                use_test = "ttest" if all_normal else "mannwhitney"
            else:
                use_test = "ttest" if all_normal else "mannwhitney"

            pairwise = compare_two_groups(
                groups_clean[keys[0]], groups_clean[keys[1]], test=use_test
            )
            mat = pd.DataFrame(
                [[1.0, pairwise["p_value"]], [pairwise["p_value"], 1.0]],
                index=keys, columns=keys,
            )
            all_results[target] = {
                "normality": normality,
                "variance_homogeneity": var_result,
                "test_path": use_test,
                "omnibus_result": None,
                "pairwise_matrix": mat,
                "pairwise_sig": {f"{keys[0]} vs {keys[1]}": pairwise},
            }
            continue

        # ── 3+ groups: auto-select parametric vs non-parametric ──
        if test_method in ("ttest", "mannwhitney"):
            # User forced a two-group test on 3+ groups — run pairwise vs control
            test_path = test_method
            omnibus = None
            sig = {}
            if control_group in groups_clean:
                ctrl = groups_clean[control_group]
                for g, vals in groups_clean.items():
                    if g == control_group:
                        continue
                    sig[f"{g} vs {control_group}"] = compare_two_groups(
                        vals, ctrl, test=test_method
                    )
            mat = pd.DataFrame(np.eye(len(group_keys)), index=group_keys, columns=group_keys)
            all_results[target] = {
                "normality": normality,
                "variance_homogeneity": var_result,
                "test_path": test_path,
                "omnibus_result": omnibus,
                "pairwise_matrix": mat,
                "pairwise_sig": sig,
            }
            continue

        # Auto or explicit anova/kruskal
        if test_method == "anova":
            parametric = True
        elif test_method == "kruskal":
            parametric = False
        elif test_method == "auto":
            parametric = all_normal and equal_var
        else:
            parametric = all_normal and equal_var

        if parametric:
            test_path = "parametric"
            # ANOVA
            arrays = [groups_clean[g] for g in groups_clean]
            f_stat, p_anova = stats.f_oneway(*arrays)
            omnibus = {
                "statistic": f_stat, "p_value": p_anova,
                "significance": significance_label(p_anova),
                "test_name": "One-way ANOVA",
                "n_groups": len(groups_clean),
            }
            # Tukey HSD full matrix
            mat = _tukey_pairwise(groups_clean)
        else:
            test_path = "nonparametric"
            # Kruskal-Wallis
            arrays = [groups_clean[g] for g in groups_clean]
            h_stat, p_kw = stats.kruskal(*arrays)
            omnibus = {
                "statistic": h_stat, "p_value": p_kw,
                "significance": significance_label(p_kw),
                "test_name": "Kruskal-Wallis",
                "n_groups": len(groups_clean),
            }
            # Dunn's test full matrix
            mat = _dunn_pairwise(groups_clean)

        # Build significance dict
        sig = {}
        labels = list(groups_clean.keys())
        for i, j in itertools.combinations(range(len(labels)), 2):
            p = mat.iloc[i, j]
            sig[f"{labels[i]} vs {labels[j]}"] = {
                "statistic": np.nan, "p_value": p,
                "significance": significance_label(p),
                "test_name": "Tukey HSD" if parametric else "Dunn's test",
            }

        all_results[target] = {
            "normality": normality,
            "variance_homogeneity": var_result,
            "test_path": test_path,
            "omnibus_result": omnibus,
            "pairwise_matrix": mat,
            "pairwise_sig": sig,
        }

    return all_results


# ── display helpers ───────────────────────────────────────────

def stats_to_dataframe(stats_results: dict) -> pd.DataFrame:
    """Convert stats results into display-ready DataFrames.

    Returns two DataFrames:
        1. omnibus_df: omnibus test results
        2. matrix_df: P-value matrix (first row = column headers)
    """
    omnibus_rows = []
    matrix_dfs = {}

    for gene, sres in stats_results.items():
        omnibus = sres.get("omnibus_result")
        if omnibus and omnibus.get("p_value") is not None:
            omnibus_rows.append({
                "目的基因": gene,
                "检验路径": sres.get("test_path", ""),
                "检验方法": omnibus.get("test_name", ""),
                "统计量": f"{omnibus['statistic']:.4f}",
                "P值": f"{omnibus['p_value']:.6f}",
                "显著性": omnibus.get("significance", ""),
                "正态性": _normality_summary(sres.get("normality", {})),
                "方差齐性": _var_summary(sres.get("variance_homogeneity", {})),
            })

        mat = sres.get("pairwise_matrix")
        if mat is not None and not mat.empty:
            matrix_dfs[gene] = mat

    omnibus_df = pd.DataFrame(omnibus_rows) if omnibus_rows else pd.DataFrame(
        {"提示": ["无统计检验结果"]}
    )
    return omnibus_df, matrix_dfs


def stats_matrix_to_html(matrix_dfs: dict) -> str:
    """Render P-value matrices as HTML tables with significance color coding.

    Red background for p<0.05, light yellow for p<0.1, green for p>=0.1.
    """
    if not matrix_dfs:
        return "<p>无 P 值矩阵</p>"

    css = """
    <style>
    .pmatrix { border-collapse: collapse; font-size: 12px; margin: 8px 0; }
    .pmatrix th { background: #4472C4; color: white; padding: 6px 10px; text-align: center; }
    .pmatrix td { padding: 5px 8px; text-align: center; border: 1px solid #ddd;
                  font-variant-numeric: tabular-nums; }
    .pmatrix .p-sig   { background: #ffcccc; font-weight: bold; }
    .pmatrix .p-trend { background: #fff8cc; }
    .pmatrix .p-ns    { background: #e6f3e6; }
    .pmatrix .p-self  { background: #f0f0f0; color: #999; }
    </style>
    """
    html = css

    for gene, mat in matrix_dfs.items():
        html += f"<h4>{gene} — P 值矩阵</h4>"
        html += '<table class="pmatrix"><thead><tr><th></th>'
        for col in mat.columns:
            html += f"<th>{col}</th>"
        html += "</tr></thead><tbody>"

        for row_label in mat.index:
            html += f"<tr><th>{row_label}</th>"
            for col_label in mat.columns:
                p = mat.loc[row_label, col_label]
                if row_label == col_label:
                    cls = "p-self"
                    text = "—"
                elif pd.isna(p):
                    cls = ""
                    text = "N/A"
                else:
                    if p < 0.05:
                        cls = "p-sig"
                    elif p < 0.1:
                        cls = "p-trend"
                    else:
                        cls = "p-ns"
                    text = f"{p:.4f}"
                html += f'<td class="{cls}">{text}</td>'
            html += "</tr>"
        html += "</table><br>"

    return html


def _normality_summary(normality: dict) -> str:
    if not normality:
        return "N/A"
    parts = []
    for g, n in normality.items():
        if n.get("is_normal"):
            parts.append(f"{g}: ✓")
        else:
            parts.append(f"{g}: ✗(p={n.get('p_value', 0):.3f})")
    return ", ".join(parts)


def _var_summary(var_result: dict) -> str:
    if not var_result or var_result.get("p_value") is None or np.isnan(var_result["p_value"]):
        return "N/A"
    p = var_result["p_value"]
    eq = var_result.get("equal_var", False)
    return f"{'✓ 齐性' if eq else '✗ 不齐'} (p={p:.3f})"


# ── chart bracket helpers ─────────────────────────────────────

def get_significance_brackets(
    stats_results: dict,
    groups: list[str],
    target_gene: str,
    group_to_x: dict[str, int] | None = None,
) -> list[dict]:
    """Extract pairwise sig comparisons and return bracket coordinates."""
    brackets = []
    if target_gene not in stats_results:
        return brackets

    sres = stats_results[target_gene]
    # Use new pairwise_sig if available, else fallback to legacy pairwise_results
    pairwise = sres.get("pairwise_sig") or sres.get("pairwise_results", {})

    for comp, detail in pairwise.items():
        if not isinstance(detail, dict):
            continue
        if not is_significant(detail.get("p_value", np.nan)):
            continue

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


# ── legacy compatibility ──────────────────────────────────────

def compare_multi_groups(
    groups: dict[str, np.ndarray],
    control_label: str | None = None,
) -> dict:
    """Legacy wrapper — delegates to run_pipeline's internal logic.
    Kept for backward compatibility with existing tests.
    """
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

    arrays = list(clean_data.values())
    f_stat, p_anova = stats.f_oneway(*arrays)
    anova = {
        "statistic": f_stat,
        "p_value": p_anova,
        "significance": significance_label(p_anova),
        "test_name": "One-way ANOVA",
        "n_groups": len(clean_data),
    }

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
