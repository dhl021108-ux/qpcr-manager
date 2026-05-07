# -*- coding: utf-8 -*-
"""Tests for the 6-step ΔΔCt pipeline with global control baseline."""

import numpy as np
import pandas as pd
import pytest

from src.calculator import compute, compute_summary, compute_full_table


def make_wells(ctrl_target=30.0, trt_target=28.0, ref_ct=20.0, noise=0.1):
    """Build per-well DataFrame with 3 wells × 2 samples per group."""
    records = []
    for grp, tg in [("Ctrl", ctrl_target), ("Exp", trt_target)]:
        for bio in range(1, 3):
            sid = f"{grp}-{bio}"
            for _ in range(3):
                records.append({
                    "sample_id": sid,
                    "group": grp,
                    "REF_ref": ref_ct + np.random.default_rng(42).uniform(-noise, noise),
                    "GENE_target": tg + np.random.default_rng(99).uniform(-noise, noise),
                })
    return pd.DataFrame(records)


def make_gene_cols():
    return [
        {"gene_name": "REF", "gene_type": "ref", "numeric_col": "REF_ref"},
        {"gene_name": "GENE", "gene_type": "target", "numeric_col": "GENE_target"},
    ]


class TestCompute:
    def test_basic_shape(self):
        df = make_wells()
        gc = make_gene_cols()
        result, per_well = compute(df, gc, ["REF"], ["GENE"], "Ctrl")
        assert len(result) == 4  # 2 bio × 2 groups
        assert len(per_well) == 12  # 4 bio × 3 wells
        assert "mean1" in per_well.columns
        assert "mean2" in per_well.columns
        assert "mean3" in per_well.columns
        assert "归一化数据" in per_well.columns

    def test_equal_groups_norm_one(self):
        """Same ΔCt in both groups → normalized ≈ 1."""
        df = pd.DataFrame({
            "sample_id": ["C1"]*3 + ["C2"]*3 + ["E1"]*3 + ["E2"]*3,
            "group": ["Ctrl"]*6 + ["Exp"]*6,
            "REF_ref": [20.0]*12,
            "GENE_target": [25.0]*12,
        })
        gc = [
            {"gene_name": "REF", "gene_type": "ref", "numeric_col": "REF_ref"},
            {"gene_name": "GENE", "gene_type": "target", "numeric_col": "GENE_target"},
        ]
        result, _ = compute(df, gc, ["REF"], ["GENE"], "Ctrl")
        assert abs(result["normalized_data"].mean() - 1.0) < 0.01

    def test_upregulation_global_baseline(self):
        """Ctrl target~30, Exp target~28: 2-cycle lower → ~4× up."""
        df = pd.DataFrame({
            "sample_id": ["C1"]*3 + ["C2"]*3 + ["T1"]*3 + ["T2"]*3,
            "group": ["Ctrl"]*6 + ["Trt"]*6,
            "REF_ref": [20.0]*12,
            "GENE_target": [30.0]*6 + [28.0]*6,
        })
        gc = [
            {"gene_name": "REF", "gene_type": "ref", "numeric_col": "REF_ref"},
            {"gene_name": "GENE", "gene_type": "target", "numeric_col": "GENE_target"},
        ]
        result, _ = compute(df, gc, ["REF"], ["GENE"], "Ctrl")
        trt_norm = result[result["group"] == "Trt"]["normalized_data"].mean()
        # ΔCt: 10→8, ΔΔCt: 0→-2, FC: 1→4
        assert abs(trt_norm - 4.0) < 0.1, f"Expected ~4.0, got {trt_norm:.3f}"

    def test_downregulation_global_baseline(self):
        """Ctrl target~30, Exp target~32: 2-cycle higher → ~0.25×."""
        df = pd.DataFrame({
            "sample_id": ["C1"]*3 + ["C2"]*3 + ["T1"]*3 + ["T2"]*3,
            "group": ["Ctrl"]*6 + ["Trt"]*6,
            "REF_ref": [20.0]*12,
            "GENE_target": [30.0]*6 + [32.0]*6,
        })
        gc = [
            {"gene_name": "REF", "gene_type": "ref", "numeric_col": "REF_ref"},
            {"gene_name": "GENE", "gene_type": "target", "numeric_col": "GENE_target"},
        ]
        result, _ = compute(df, gc, ["REF"], ["GENE"], "Ctrl")
        trt_norm = result[result["group"] == "Trt"]["normalized_data"].mean()
        assert abs(trt_norm - 0.25) < 0.01, f"Expected ~0.25, got {trt_norm:.3f}"

    def test_missing_value_is_nan(self):
        df = pd.DataFrame({
            "sample_id": ["A"]*3 + ["B"]*3 + ["C"]*3,
            "group": ["Ctrl"]*6 + ["Exp"]*3,
            "REF_ref": [20.0]*9,
            "GENE_target": [25.0, 25.0, np.nan, 25, 25, 25, 25, 25, 25],
        })
        gc = [
            {"gene_name": "REF", "gene_type": "ref", "numeric_col": "REF_ref"},
            {"gene_name": "GENE", "gene_type": "target", "numeric_col": "GENE_target"},
        ]
        result, _ = compute(df, gc, ["REF"], ["GENE"], "Ctrl")
        sample_A = result[result["sample_id"] == "A"]
        assert pd.notna(sample_A["delta_ct"].values[0])

    def test_per_group_mean3(self):
        """mean3 should be per-group, not a single global value."""
        df = pd.DataFrame({
            "sample_id": ["C1"]*3 + ["C2"]*3 + ["T1"]*3 + ["T2"]*3,
            "group": ["Ctrl"]*6 + ["Trt"]*6,
            "REF_ref": [20.0]*12,
            "GENE_target": [30.0]*6 + [28.0]*6,
        })
        gc = [
            {"gene_name": "REF", "gene_type": "ref", "numeric_col": "REF_ref"},
            {"gene_name": "GENE", "gene_type": "target", "numeric_col": "GENE_target"},
        ]
        result, per_well = compute(df, gc, ["REF"], ["GENE"], "Ctrl")
        ctrl_mean3 = per_well[per_well["group"] == "Ctrl"]["mean3"].iloc[0]
        trt_mean3 = per_well[per_well["group"] == "Trt"]["mean3"].iloc[0]
        # Ctrl mean3 ≈ 1.0, Trt mean3 ≈ 4.0
        assert abs(ctrl_mean3 - 1.0) < 0.01
        assert abs(trt_mean3 - 4.0) < 0.1
        # Different groups have different mean3
        assert ctrl_mean3 != trt_mean3

    def test_normalized_uses_control_mean3(self):
        """Normalized = mean2 / CONTROL's mean3 (not own group's mean3)."""
        df = pd.DataFrame({
            "sample_id": ["C1"]*3 + ["C2"]*3 + ["T1"]*3 + ["T2"]*3,
            "group": ["Ctrl"]*6 + ["Trt"]*6,
            "REF_ref": [20.0]*12,
            "GENE_target": [30.0]*6 + [28.0]*6,
        })
        gc = [
            {"gene_name": "REF", "gene_type": "ref", "numeric_col": "REF_ref"},
            {"gene_name": "GENE", "gene_type": "target", "numeric_col": "GENE_target"},
        ]
        result, per_well = compute(df, gc, ["REF"], ["GENE"], "Ctrl")
        ctrl_mean3 = per_well[per_well["group"] == "Ctrl"]["mean3"].iloc[0]
        # Each sample's normalized = mean2 / ctrl_mean3
        for _, r in result.iterrows():
            expected = r["mean2"] / ctrl_mean3
            assert abs(r["normalized_data"] - expected) < 0.001

    def test_compute_full_table(self):
        df = pd.DataFrame({
            "Sample": ["S1"]*3 + ["S2"]*3,
            "Group": ["Ctrl"]*3 + ["Exp"]*3,
            "GAPDH Ct": [20.0]*6,
            "IL6 Ct": [25.0]*3 + [23.0]*3,
        })
        full, result = compute_full_table(
            df, ref_col="GAPDH Ct", target_col="IL6 Ct",
            control_group="Ctrl", sample_col="Sample", group_col="Group",
        )
        assert "mean3" in full.columns
        assert "归一化数据" in full.columns
        assert len(full) == 6
        assert len(result) == 2
        # mean3 per group
        ctrl_m3 = full[full["分组"] == "Ctrl"]["mean3"].iloc[0]
        exp_m3 = full[full["分组"] == "Exp"]["mean3"].iloc[0]
        assert ctrl_m3 != exp_m3


class TestEdgeCases:
    def test_invalid_control_group(self):
        df = make_wells()
        with pytest.raises(ValueError):
            compute(df, make_gene_cols(), ["REF"], ["GENE"], "BadGroup")

    def test_no_ref_genes(self):
        with pytest.raises(ValueError):
            compute(make_wells(), make_gene_cols(), [], ["GENE"], "Ctrl")

    def test_no_target_genes(self):
        with pytest.raises(ValueError):
            compute(make_wells(), make_gene_cols(), ["REF"], [], "Ctrl")
