# -*- coding: utf-8 -*-
"""Tests for data parsing module."""

from io import BytesIO

import pandas as pd
import pytest

from src.parser import parse_uploaded_file, detect_column_type


class TestDetectColumnType:
    def test_sample_id_chinese(self):
        role, _, _ = detect_column_type("样本名称")
        assert role == "sample_id"

    def test_sample_id_english(self):
        role, _, _ = detect_column_type("sample_id")
        assert role == "sample_id"

    def test_group_chinese(self):
        role, _, _ = detect_column_type("分组")
        assert role == "group"

    def test_group_english(self):
        role, _, _ = detect_column_type("group")
        assert role == "group"

    def test_analysis_type_as_group(self):
        role, _, _ = detect_column_type("总分析")
        assert role == "group"

    def test_ref_gene(self):
        role, name, gtype = detect_column_type("GAPDH(内参)")
        assert role == "gene"
        assert name == "GAPDH"
        assert gtype == "ref"

    def test_target_gene(self):
        role, name, gtype = detect_column_type("IL-6(目的)")
        assert role == "gene"
        assert name == "IL-6"
        assert gtype == "target"

    def test_generic_ref_gene(self):
        role, name, gtype = detect_column_type("内参基因")
        assert role == "gene"
        assert gtype == "ref"

    def test_generic_target_gene(self):
        role, name, gtype = detect_column_type("目的基因")
        assert role == "gene"
        assert gtype == "target"

    def test_generic_ref_en(self):
        role, name, gtype = detect_column_type("Ref Gene")
        assert role == "gene"
        assert gtype == "ref"

    def test_unknown_gene(self):
        role, name, gtype = detect_column_type("TNF-a")
        assert role == "gene"
        assert gtype == "unknown"

    def test_note_column(self):
        role, _, _ = detect_column_type("备注")
        assert role == "note"


class TestParseUploadedFile:
    def _make_csv(self, content):
        bio = BytesIO(content.encode("utf-8"))
        bio.name = "test.csv"
        return bio

    def test_valid_csv(self):
        csv = "sample_name,group,GAPDH(ref),IL-6(target),notes\nCtrl-1,Control,18.5,28.3,\nCtrl-2,Control,18.7,28.1,\n"
        bio = self._make_csv(csv)
        df, meta = parse_uploaded_file(bio)
        assert meta["n_samples"] == 2
        assert meta["n_genes"] == 2
        assert len(meta["ref_genes"]) == 1
        assert len(meta["target_genes"]) == 1
        assert df.loc[0, "GAPDH_ref"] == 18.5
        assert df.loc[0, "IL-6_target"] == 28.3

    def test_missing_values_flag_warning(self):
        csv = "sample_name,group,GAPDH(ref),IL-6(target)\nCtrl-1,Control,18.5,\nCtrl-2,Control,18.7,28.1\n"
        bio = self._make_csv(csv)
        _, meta = parse_uploaded_file(bio)
        assert len(meta["warnings"]) > 0

    def test_technical_replicates_allowed(self):
        """Same sample ID appearing multiple times should NOT be an error (technical replicates)."""
        csv = (
            "sample_name,group,GAPDH(ref),IL-6(target)\n"
            "Ctrl-1,Control,18.5,28.3\n"
            "Ctrl-1,Control,18.7,28.1\n"
            "Ctrl-1,Control,18.6,28.2\n"
        )
        bio = self._make_csv(csv)
        df, meta = parse_uploaded_file(bio)
        # Should not flag as error — these are technical replicates
        assert len(meta["errors"]) == 0
        assert meta["n_samples"] == 1  # 1 unique sample
        assert meta["n_rows"] == 3  # 3 rows (wells)

    def test_exact_duplicates_warning(self):
        """Exact duplicate rows (same sample, group, AND Ct values) should warn."""
        csv = (
            "sample_name,group,GAPDH(ref),IL-6(target)\n"
            "Ctrl-1,Control,18.5,28.3\n"
            "Ctrl-1,Control,18.5,28.3\n"
        )
        bio = self._make_csv(csv)
        _, meta = parse_uploaded_file(bio)
        assert len(meta["warnings"]) > 0
        assert any("duplicate" in w.lower() for w in meta["warnings"])

    def test_no_annotations_all_unknown(self):
        csv = "sample_name,group,GAPDH,IL-6\nCtrl-1,Control,18.5,28.3\n"
        bio = self._make_csv(csv)
        _, meta = parse_uploaded_file(bio)
        assert len(meta["unknown_genes"]) == 2
        assert len(meta["ref_genes"]) == 0
        assert len(meta["target_genes"]) == 0

    def test_negative_ct_value_error(self):
        csv = "sample_name,group,GAPDH(ref),IL-6(target)\nCtrl-1,Control,-5.0,28.3\n"
        bio = self._make_csv(csv)
        _, meta = parse_uploaded_file(bio)
        assert any("<= 0" in e for e in meta["errors"])

    def test_high_ct_warning(self):
        csv = "sample_name,group,GAPDH(ref),IL-6(target)\nCtrl-1,Control,18.5,38.0\n"
        bio = self._make_csv(csv)
        _, meta = parse_uploaded_file(bio)
        assert any("> 35" in w for w in meta["warnings"])

    def test_forward_fill_sample_id(self):
        """Blank sample_id cells should inherit from the row above (merged Excel cells)."""
        csv = (
            "sample_name,group,GAPDH(ref),IL-6(target)\n"
            "Ctrl-1,Control,18.5,28.3\n"
            ",,18.7,28.1\n"
            ",,18.6,28.2\n"
        )
        bio = self._make_csv(csv)
        df, meta = parse_uploaded_file(bio)
        assert meta["n_samples"] == 1  # all are Ctrl-1
        assert meta["n_rows"] == 3  # 3 wells
        assert (df["sample_id"] == "Ctrl-1").all()

    def test_forward_fill_group(self):
        """Blank group cells should inherit from above."""
        csv = (
            "sample_name,group,GAPDH(ref),IL-6(target)\n"
            "S1,Control,18.5,28.3\n"
            "S2,,18.7,28.1\n"
        )
        bio = self._make_csv(csv)
        df, meta = parse_uploaded_file(bio)
        assert df.loc[1, "group"] == "Control"
