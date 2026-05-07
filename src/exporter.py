"""结果导出：Excel / CSV / 图表图片 / PDF 报告"""

from io import BytesIO

import pandas as pd
import plotly.graph_objects as go

try:
    from fpdf import FPDF

    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False


def to_excel_bytes(
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
    stats_results: dict | None = None,
) -> bytes:
    """Export results as a multi-sheet Excel workbook.

    Sheets: 原始数据 | 详细结果 | 分组汇总 | 统计检验
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if raw_df is not None:
            raw_df.to_excel(writer, sheet_name="原始数据", index=False)
        result_df.to_excel(writer, sheet_name="详细结果", index=False)
        summary_df.to_excel(writer, sheet_name="分组汇总", index=False)

        if stats_results:
            stats_rows = []
            for target, sres in stats_results.items():
                pairwise = sres.get("pairwise_results", {})
                for comp, detail in pairwise.items():
                    if isinstance(detail, dict):
                        stats_rows.append(
                            {
                                "目的基因": target,
                                "比较": comp,
                                "检验方法": detail.get("test_name", ""),
                                "统计量": detail.get("statistic", ""),
                                "p值": detail.get("p_value", ""),
                                "显著性": detail.get("significance", ""),
                            }
                        )
            if stats_rows:
                stats_df = pd.DataFrame(stats_rows)
                stats_df.to_excel(writer, sheet_name="统计检验", index=False)

    return output.getvalue()


def to_csv_bytes(result_df: pd.DataFrame) -> bytes:
    """Export as CSV."""
    return result_df.to_csv(index=False).encode("utf-8-sig")


# Chart export removed — use Plotly modebar camera button (frontend)
# for PNG/SVG/PDF downloads. No kaleido/to_image dependency.
def to_pdf_report(
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stats_results: dict,
    figs: dict[str, go.Figure],
    experiment_name: str = "qPCR 实验",
) -> bytes:
    """Generate a PDF report with methods, results table, stats, and charts.

    Args:
        result_df: Per-sample results
        summary_df: Grouped summary
        stats_results: Statistics output
        figs: Dict of {name: go.Figure} for charts
        experiment_name: Title for the report
    """
    if not HAS_FPDF:
        raise ImportError("fpdf2 未安装，无法生成 PDF 报告。请运行: pip install fpdf2")

    pdf = FPDF()
    pdf.add_page()

    # 标题
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, experiment_name, ln=True, align="C")
    pdf.ln(5)

    # 方法说明
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. 计算方法", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "采用 2^(-DeltaDeltaCt) 方法 (Livak & Schmittgen, 2001) 计算相对表达量。\n"
        "DeltaCt = Ct(目的基因) - Ct(内参基因)\n"
        "DeltaDeltaCt = DeltaCt(实验组) - mean(DeltaCt(对照组))\n"
        "Fold Change = 2^(-DeltaDeltaCt)\n"
        "统计检验使用 SciPy 的 Student's t-test / One-way ANOVA。"
    )
    pdf.ln(5)

    # 汇总表
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. 分组汇总", ln=True)
    pdf.set_font("Helvetica", "", 9)

    cols = ["target_gene", "group", "n", "fc_mean", "fc_sem", "delta_delta_ct_mean"]
    col_labels = ["目的基因", "分组", "n", "FC均值", "FC SEM", "DeltaDeltaCt"]
    col_widths = [30, 30, 12, 28, 28, 30]

    for label, w in zip(col_labels, col_widths):
        pdf.cell(w, 6, label, border=1)
    pdf.ln()

    for _, row in summary_df.iterrows():
        vals = [
            str(row.get("target_gene", "")),
            str(row.get("group", "")),
            str(int(row.get("n", 0))),
            f"{row.get('fc_mean', 0):.3f}",
            f"{row.get('fc_sem', 0):.3f}",
            f"{row.get('delta_delta_ct_mean', 0):.3f}",
        ]
        for v, w in zip(vals, col_widths):
            pdf.cell(w, 6, v, border=1)
        pdf.ln()

    pdf.ln(5)

    # 统计
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. 统计检验", ln=True)
    pdf.set_font("Helvetica", "", 9)

    for target, sres in stats_results.items():
        pdf.set_font("Helvetica", "B", 9)
        pdf.cell(0, 6, f"基因: {target}", ln=True)
        pdf.set_font("Helvetica", "", 9)

        pairwise = sres.get("pairwise_results", {})
        for comp, detail in pairwise.items():
            if isinstance(detail, dict):
                line = (
                    f"  {comp}: {detail.get('test_name', '')}, "
                    f"p = {detail.get('p_value', np.nan):.4f} {detail.get('significance', '')}"
                )
                pdf.cell(0, 5, line, ln=True)
        pdf.ln(2)

    # 图表占位 — 由于 fpdf2 不支持直接嵌入 Plotly 矢量图，
    # 提示用户单独导出 PNG
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "4. 图表", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5,
        "图表已单独导出为 PNG/SVG 文件（300 dpi），适合论文发表。\n"
        "请在应用中点击「导出图表 PNG」按钮下载。"
    )

    return pdf.output(dest="S").encode("latin-1")


import numpy as np
