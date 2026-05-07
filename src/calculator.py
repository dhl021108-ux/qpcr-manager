# -*- coding: utf-8 -*-
"""Core calculation: 6-step per-well ΔΔCt pipeline.

Pipeline (per-well, step by step):
  1. Per-well ΔCt = Ct(target) - Ct(ref);
     mean1 = AVG(ΔCt of 3 wells of same sample)
  2. Global control baseline = AVG(all CONTROL samples' mean1)  (scalar)
  3. Per-well ΔΔCt = ΔCt - global_baseline
  4. Per-well 2^(-ΔΔCt);
     mean2 = AVG(2^(-ΔΔCt) of 3 wells of same sample)
  5. mean3 per group = AVG(all samples' mean2 within that group)
  6. Normalized = mean2 / CONTROL's mean3  (for all samples)
"""

import numpy as np
import pandas as pd
from scipy.stats import gmean


def compute(
    df: pd.DataFrame,
    gene_cols: list,
    ref_gene_names: list,
    target_gene_names: list,
    control_group: str,
):
    """Run the full 6-step per-well ΔΔCt calculation pipeline.

    Returns:
        (result, per_well) tuple:
        - result: per-biological-sample DataFrame
        - per_well: per-well DataFrame with all intermediate columns
    """
    if not ref_gene_names:
        raise ValueError("至少需要 1 个内参基因")
    if not target_gene_names:
        raise ValueError("至少需要 1 个目的基因")
    if control_group not in df["group"].values:
        raise ValueError(
            f"对照组 '{control_group}' 不在分组标签中: "
            f"{df['group'].unique().tolist()}"
        )

    name_to_col = _build_name_map(gene_cols)

    # Build combined ref Ct (geometric mean for multi-ref)
    ref_cols = [name_to_col[n] for n in ref_gene_names if n in name_to_col]
    ref_cols = [c for c in ref_cols if c in df.columns]
    if not ref_cols:
        raise ValueError(f"未找到内参基因列: {ref_gene_names}")

    if len(ref_cols) == 1:
        df["_ref_ct"] = pd.to_numeric(df[ref_cols[0]], errors="coerce")
    else:
        ref_vals = df[ref_cols].values.astype(float)
        with np.errstate(invalid="ignore"):
            df["_ref_ct"] = gmean(ref_vals, axis=1, nan_policy="omit")

    # ==== Step 1: Per-well ΔCt + sample mean1 ====
    rows = []
    for tg_name in target_gene_names:
        tg_col = name_to_col.get(tg_name)
        if tg_col is None or tg_col not in df.columns:
            continue

        ct_target_series = pd.to_numeric(df[tg_col], errors="coerce")
        ct_ref_series = df["_ref_ct"]

        for idx in df.index:
            ct_t = ct_target_series[idx]
            ct_r = ct_ref_series[idx]
            sid = df.at[idx, "sample_id"]
            grp = df.at[idx, "group"]

            delta_ct = (ct_t - ct_r) if (pd.notna(ct_t) and pd.notna(ct_r)) else np.nan

            rows.append({
                "sample_id": sid,
                "group": grp,
                "target_gene": tg_name,
                "ref_gene": "+".join(ref_gene_names),
                "ct_target": ct_t,
                "ct_ref": ct_r,
                "delta_ct": delta_ct,
            })

    per_well = pd.DataFrame(rows)
    if per_well.empty:
        raise ValueError("未产生任何结果 — 请检查基因列映射")

    # mean1 = AVG ΔCt per sample (same value on all 3 wells of that sample)
    mean1_map = (
        per_well.groupby(["sample_id", "group", "target_gene"])["delta_ct"]
        .mean().to_dict()
    )
    per_well["mean1"] = per_well.apply(
        lambda r: mean1_map.get(
            (r["sample_id"], r["group"], r["target_gene"]), np.nan
        ), axis=1,
    )

    # ==== Step 2: Global control baseline = AVG(all CONTROL samples' mean1) ====
    # Extract one mean1 per CONTROL sample
    ctrl_per_well = per_well[per_well["group"] == control_group]
    ctrl_mean1s = ctrl_per_well.groupby("sample_id")["mean1"].first()
    global_baseline = ctrl_mean1s.mean()  # single scalar

    if pd.isna(global_baseline):
        raise ValueError(
            f"无法计算对照组 '{control_group}' 的全局基准值 — 请检查 Ct 数据"
        )

    # ==== Step 3: Per-well ΔΔCt = ΔCt - global_baseline ====
    per_well["delta_delta_ct"] = per_well["delta_ct"] - global_baseline

    # ==== Step 4: Per-well 2^(-ΔΔCt) + sample mean2 ====
    per_well["fc_per_well"] = np.power(2.0, -per_well["delta_delta_ct"])

    mean2_map = (
        per_well.groupby(["sample_id", "group", "target_gene"])["fc_per_well"]
        .mean().to_dict()
    )
    per_well["mean2"] = per_well.apply(
        lambda r: mean2_map.get(
            (r["sample_id"], r["group"], r["target_gene"]), np.nan
        ), axis=1,
    )

    # ==== Step 5: mean3 per group = AVG(all samples' mean2 in that group) ====
    mean3_map = {}
    for grp in per_well["group"].unique():
        grp_mean2s = per_well[per_well["group"] == grp].groupby("sample_id")["mean2"].first()
        mean3_map[grp] = grp_mean2s.mean()

    per_well["mean3"] = per_well["group"].map(mean3_map)
    control_mean3 = mean3_map.get(control_group, np.nan)

    # ==== Step 6: Normalized = mean2 / CONTROL's mean3 ====
    if pd.notna(control_mean3) and control_mean3 > 0:
        per_well["归一化数据"] = per_well["mean2"] / control_mean3
    else:
        per_well["归一化数据"] = np.nan

    # ---- Build per-sample result ----
    result_rows = []
    for (sid, grp, tg), grp_data in per_well.groupby(
        ["sample_id", "group", "target_gene"]
    ):
        result_rows.append({
            "sample_id": sid,
            "group": grp,
            "target_gene": tg,
            "ref_gene": "+".join(ref_gene_names),
            "ct_target_mean": grp_data["ct_target"].mean(),
            "ct_ref_mean":    grp_data["ct_ref"].mean(),
            "delta_ct":       grp_data["delta_ct"].mean(),
            "mean1":          grp_data["mean1"].iloc[0],
            "delta_delta_ct": grp_data["delta_delta_ct"].mean(),
            "fold_change":    grp_data["fc_per_well"].mean(),  # raw mean2
            "mean2":          grp_data["mean2"].iloc[0],
            "mean3":          grp_data["mean3"].iloc[0],
            "normalized_data": (
                grp_data["mean2"].iloc[0] / control_mean3
                if pd.notna(control_mean3) and control_mean3 > 0
                else np.nan
            ),
        })

    result = pd.DataFrame(result_rows)
    result["log2_fc"] = np.log2(
        result["normalized_data"].replace(0, np.nan)
    )
    # For chart compatibility: fold_change = normalized_data
    result["fold_change"] = result["normalized_data"]

    return result, per_well


def compute_full_table(
    df: pd.DataFrame,
    ref_col: str,
    target_col: str,
    control_group: str,
    sample_col: str = "Sample",
    group_col: str = "Group",
):
    """Run 6-step pipeline and return a per-well table with ALL intermediate columns.

    Column order matches input row order exactly.
    """
    ref_numeric = ref_col + "_ref"
    target_numeric = target_col + "_target"
    gene_cols = [
        {"col_name": ref_col, "gene_name": ref_col, "gene_type": "ref",
         "numeric_col": ref_numeric},
        {"col_name": target_col, "gene_name": target_col, "gene_type": "target",
         "numeric_col": target_numeric},
    ]

    work_df = df.rename(columns={
        sample_col: "sample_id",
        group_col: "group",
        ref_col: ref_numeric,
        target_col: target_numeric,
    })

    result, per_well = compute(
        work_df, gene_cols, [ref_col], [target_col], control_group,
    )

    # Restore display column names (preserving row order)
    per_well["样本"] = per_well["sample_id"]
    per_well["分组"] = per_well["group"]

    return per_well, result


def compute_summary(result_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-group summary statistics using normalized_data."""
    def _sem(x):
        if len(x) < 2:
            return np.nan
        return np.std(x, ddof=1) / np.sqrt(len(x))

    summary = result_df.groupby(
        ["target_gene", "group"], as_index=False
    ).agg(
        n=("normalized_data", "count"),
        fc_mean=("normalized_data", "mean"),
        fc_sd=("normalized_data", lambda x: (
            np.std(x, ddof=1) if len(x) > 1 else np.nan
        )),
        fc_sem=("normalized_data", _sem),
        delta_ct_mean=("delta_ct", "mean"),
        delta_ct_sem=("delta_ct", _sem),
        delta_delta_ct_mean=("delta_delta_ct", "mean"),
        delta_delta_ct_sem=("delta_delta_ct", _sem),
    )

    summary["log2_fc"] = np.log2(summary["fc_mean"].replace(0, np.nan))
    return summary


def _build_name_map(gene_cols: list) -> dict:
    return {gc["gene_name"]: gc["numeric_col"] for gc in gene_cols}
