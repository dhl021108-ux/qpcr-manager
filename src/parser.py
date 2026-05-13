# -*- coding: utf-8 -*-
"""CT data parser: Excel/CSV -> validated DataFrame + metadata.

Supports:
  - Gene column headers with (内参)/(目的) annotations (e.g. "GAPDH(内参)")
  - Generic gene column headers (e.g. "内参基因", "目的基因", "Ref Gene", "Target Gene")
  - Technical replicates: same sample_id appearing in multiple rows (no longer an error)
  - Forward-fill for merged cells (blank sample_id/group inherit from row above)
"""

import re

import numpy as np
import pandas as pd


def detect_column_type(col_name: str) -> tuple:
    """Parse a column header to determine its role and gene type.

    Returns (role, gene_name_or_None, gene_type_or_None).
    """
    c = str(col_name).strip()

    # Sample ID columns
    if c.lower() in {"sample_id", "sample", "sample name", "sample_name",
                     "样品", "样本名称", "样品名称", "样本", "样品名"}:
        return ("sample_id", None, None)

    # Group columns (broad set of Chinese/English labels)
    if c.lower() in {
        "group", "treatment", "condition",
        "分组", "组别", "处理组", "实验分组", "组",
        "总分析", "总分组", "分析类别", "类别", "实验组", "组名", "组别名称",
    }:
        return ("group", None, None)

    # Note columns
    if c.lower() in {"note", "comment", "notes", "comments",
                     "备注", "注释", "说明"}:
        return ("note", None, None)

    # Pre-computed / analysis columns that should NOT be treated as gene data
    # These are often present in template files with formulas
    if _is_calculation_column(c):
        return ("ignore", None, None)

    # Annotated gene columns: "NAME(内参)" or "NAME(目的)"
    ref_match = re.match(r"^(.+)\(内参\)$", c)
    if ref_match:
        return ("gene", ref_match.group(1).strip(), "ref")

    target_match = re.match(r"^(.+)\(目的\)$", c)
    if target_match:
        return ("gene", target_match.group(1).strip(), "target")

    # English annotations
    ref_match_en = re.match(r"^(.+)\((ref|reference|housekeeping)\)$", c, re.IGNORECASE)
    if ref_match_en:
        return ("gene", ref_match_en.group(1).strip(), "ref")

    target_match_en = re.match(r"^(.+)\((target|goi)\)$", c, re.IGNORECASE)
    if target_match_en:
        return ("gene", target_match_en.group(1).strip(), "target")

    # Generic gene type headers — the header itself indicates the gene role
    generic_ref = {"ref gene", "reference gene", "ref_gene", "reference_gene",
                   "内参基因", "内参", "reference", "ref",
                   "refgene", "referencegene", "housekeeping gene"}
    if c.lower() in generic_ref:
        return ("gene", "RefGene", "ref")

    generic_target = {"target gene", "target_gene", "target",
                      "目的基因", "目的", "goi", "gene of interest",
                      "targetgene", "goi_gene"}
    if c.lower() in generic_target:
        return ("gene", "TargetGene", "target")

    # Looks like a gene name (alphanumeric with dash/dot/underscore/slash/greek)
    if re.match(r"^[\w\-\./]+$", c, re.UNICODE):
        return ("gene", c, "unknown")

    return ("unknown", c, None)


def _is_calculation_column(col_name: str) -> bool:
    """Return True if the column header looks like a pre-computed analysis column."""
    c = str(col_name).strip().lower()
    calc_patterns = [
        r"^Δct$",                # Δct
        r"^ΔΔct$",         # ΔΔct
        r"^2\^\(-ΔΔct\)$",       # 2^(-ΔΔct)
        r"^2\^\(-ddct\)$",       # 2^(-ddCt) English variant
        r"^fold.change$",
        r"^log2.*fc$",
        r"^fc$",
        r"^mean[0-9]*$",         # mean1, mean2, mean3
        r"^avg.*$",              # avg, average
        r"^sem$",
        r"^stdev$",
        r"^p.?value$",
        r"^归一化.*",             # normalized data
        r"^标准化.*",             # standardized data
    ]
    for pat in calc_patterns:
        if re.match(pat, c, re.IGNORECASE):
            return True
    # Detect any column containing Greek Delta (Δ) — typically pre-computed
    if 'Δ' in c:  # Δ
        return True
    return False


def parse_uploaded_file(uploaded_file, skip_rows: int = 0) -> tuple:
    """Parse an uploaded Excel/CSV file into a validated DataFrame with metadata.

    Returns:
        (parsed_df, meta):
          parsed_df — columns: sample_id, group, [numeric_gene_cols...], note
          meta — dict with gene_cols, n_samples, n_rows, n_genes, groups, errors, warnings
    """
    import io
    filename = getattr(uploaded_file, 'name', None)
    # If it's a string path (not a file object), open it
    if isinstance(uploaded_file, str):
        filename = uploaded_file
        uploaded_file = open(uploaded_file, 'rb')
        _should_close = True
    else:
        _should_close = False

    if filename is None:
        filename = 'data.csv'

    if hasattr(filename, 'lower'):
        filename = filename.lower()
    else:
        filename = str(filename).lower()

    if filename.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file, skiprows=skip_rows, dtype=str)
    else:
        raw_df = pd.read_excel(uploaded_file, skiprows=skip_rows, dtype=str)

    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    if raw_df.empty or len(raw_df.columns) < 2:
        raise ValueError("File format error: need at least 2 columns (sample + 1 gene)")

    # ---- Classify columns ----
    sample_col = None
    group_col = None
    note_col = None
    gene_cols: list[dict] = []
    unknown_cols: list[str] = []

    for col in raw_df.columns:
        result = detect_column_type(col)
        role = result[0]

        if role == "sample_id" and sample_col is None:
            sample_col = col
        elif role == "group" and group_col is None:
            group_col = col
        elif role == "note" and note_col is None:
            note_col = col
        elif role == "ignore":
            pass  # skip pre-computed columns
        elif role == "gene":
            _, gene_name, gene_type = result
            gene_cols.append({"col_name": col, "gene_name": gene_name, "gene_type": gene_type})
        elif role == "unknown":
            unknown_cols.append(col)

    # ---- Heuristic: if no group column found, look for categorical columns in data ----
    if group_col is None:
        for candidate in raw_df.columns[:min(3, len(raw_df.columns))]:
            if candidate == sample_col:
                continue
            vals = raw_df[candidate].dropna().str.strip()
            n_unique = vals.nunique()
            n_rows = len(vals)
            # A group column has few unique values relative to row count
            if n_unique >= 2 and n_unique <= max(10, n_rows // 3):
                group_col = candidate
                # Remove from gene/unknown lists if it was classified there
                gene_cols[:] = [gc for gc in gene_cols if gc["col_name"] != candidate]
                if candidate in unknown_cols:
                    unknown_cols.remove(candidate)
                break

    # ---- Fallbacks ----
    if sample_col is None:
        sample_col = raw_df.columns[0]
    if group_col is None and len(raw_df.columns) >= 2:
        candidate = raw_df.columns[1]
        unique_vals = raw_df[candidate].dropna().str.strip().unique()
        if len(unique_vals) <= max(10, len(raw_df) // 3):
            group_col = candidate

    # ---- Build parsed DataFrame (same index as raw_df) ----
    parsed = pd.DataFrame(index=raw_df.index)

    # Sample ID
    parsed["sample_id"] = raw_df[sample_col].astype(str).str.strip()
    parsed["sample_id"] = parsed["sample_id"].replace(
        {"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})

    # Group
    if group_col:
        parsed["group"] = raw_df[group_col].astype(str).str.strip()
        parsed["group"] = parsed["group"].replace(
            {"": np.nan, "nan": np.nan, "None": np.nan, "NaN": np.nan})

    # Note
    if note_col:
        parsed["note"] = raw_df[note_col].astype(str).str.strip()
        parsed["note"] = parsed["note"].replace({"": np.nan, "nan": np.nan})

    # ---- Forward-fill (merged Excel cells) ----
    parsed["sample_id"] = parsed["sample_id"].ffill()
    if group_col and "group" in parsed.columns:
        parsed["group"] = parsed["group"].ffill()

    # ---- Convert gene columns to numeric (before dropping rows) ----
    for gc in gene_cols:
        col = gc["col_name"]
        numeric_col = f"{gc['gene_name']}_{gc['gene_type']}"
        parsed[numeric_col] = pd.to_numeric(raw_df[col], errors="coerce")
        gc["numeric_col"] = numeric_col

    # ---- Drop rows where sample_id is still NaN ----
    n_before = len(parsed)
    parsed = parsed.dropna(subset=["sample_id"])
    n_dropped = n_before - len(parsed)
    parsed = parsed.reset_index(drop=True)

    # ---- Validate ----
    errors, warnings = _validate(parsed, gene_cols)
    if n_dropped > 0:
        warnings.insert(0, f"Dropped {n_dropped} row(s) with empty sample name")

    meta = {
        "sample_col": sample_col,
        "group_col": group_col,
        "note_col": note_col,
        "gene_cols": gene_cols,
        "unknown_cols": unknown_cols,
        "n_samples": parsed["sample_id"].nunique(),
        "n_rows": len(parsed),
        "n_genes": len(gene_cols),
        "gene_names": [gc["gene_name"] for gc in gene_cols],
        "ref_genes": [gc for gc in gene_cols if gc["gene_type"] == "ref"],
        "target_genes": [gc for gc in gene_cols if gc["gene_type"] == "target"],
        "unknown_genes": [gc for gc in gene_cols if gc["gene_type"] == "unknown"],
        "groups": (parsed["group"].dropna().unique().tolist()
                   if ("group" in parsed.columns) else []),
        "errors": errors,
        "warnings": warnings,
    }

    return parsed, meta


def _validate(df: pd.DataFrame, gene_cols: list[dict]) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []

    # Sample names non-empty
    if df["sample_id"].isna().any():
        errors.append("Some sample names are empty after forward-fill")

    # Exact duplicate rows (same sample + group + ALL gene values) — possible copy-paste error
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        warnings.append(f"Found {n_dup} exact duplicate row(s) — check for accidental copy-paste")

    # Group diversity
    if "group" in df.columns:
        unique_groups = df["group"].dropna().unique()
        if len(unique_groups) < 2:
            warnings.append("Only 1 group label found; inter-group statistics unavailable")
        if df["group"].isna().any():
            warnings.append("Some samples are missing group labels after forward-fill")

    # Technical replicate count per sample
    sample_counts = df.groupby("sample_id").size()
    tech_rep_samples = sample_counts[sample_counts > 1]
    if len(tech_rep_samples) > 0:
        rep_info = ", ".join([f"{s}(n={c})" for s, c in tech_rep_samples.items()])
        warnings.append(f"Technical replicates detected: {rep_info}")

    # Gene validations
    for gc in gene_cols:
        ncol = gc["numeric_col"]
        if ncol not in df.columns:
            continue
        vals = pd.to_numeric(df[ncol], errors="coerce")
        n_missing = int(vals.isna().sum())
        if n_missing > 0:
            idxs = df["sample_id"][vals.isna()].tolist()
            warnings.append(f"Gene {gc['gene_name']}: missing CT value(s) in sample(s) {idxs}")
        valid = vals.dropna()
        if len(valid) > 0:
            if (valid <= 0).any():
                bad = df.loc[valid[valid <= 0].index, "sample_id"].tolist()
                errors.append(f"Gene {gc['gene_name']}: CT value <= 0 in sample(s) {bad}")
            if (valid > 35).any():
                high = df.loc[valid[valid > 35].index, "sample_id"].tolist()
                warnings.append(f"Gene {gc['gene_name']}: CT > 35 in sample(s) {high}")

    return errors, warnings


def get_gene_value_columns(df: pd.DataFrame, gene_cols: list[dict]) -> list[str]:
    """Return the numeric column names for gene CT values."""
    return [gc["numeric_col"] for gc in gene_cols]


def get_sample_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return {group_label: [unique_sample_id, ...]} mapping."""
    if "group" not in df.columns:
        return {}
    return df.groupby("group")["sample_id"].unique().apply(list).to_dict()
