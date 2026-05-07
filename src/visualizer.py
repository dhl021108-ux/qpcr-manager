# -*- coding: utf-8 -*-
"""GraphPad Prism-style Plotly charts — 600x600, academic palettes, solid dots.

Features:
  - 600x600 square canvas
  - 10 muted academic palettes (Cell / Nature / PNAS style)
  - Solid black data points
  - Black SEM error bars
  - Auto significance brackets with stars
  - SVG / PNG / PDF export
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.statistics import get_significance_brackets

# ── 10 muted academic color palettes ───────────────────────

PALETTES = {
    "Cell 期刊 (Cell)": [
        "#B3C6E7", "#F4B4C2", "#B4D5B4", "#E8D4A2",
        "#C4B8D5", "#F2C8A5", "#A8D8D8", "#E0C8C8",
    ],
    "Nature 期刊 (Nature)": [
        "#E8A5A5", "#A5C8E8", "#A5D8C8", "#D8C8A5",
        "#C8A5D0", "#E8D0A5", "#A5D0D0", "#D0B8B8",
    ],
    "Science 期刊 (Science)": [
        "#B8D0E8", "#E8C0C0", "#B8D8C8", "#E0D8B0",
        "#C8B8E0", "#E8D0B8", "#A0C8C8", "#D8C0C0",
    ],
    "莫兰迪蓝灰 (Morandi Blue)": [
        "#8FA8BF", "#A8B8C8", "#98A8B8", "#B0BCC8",
        "#8898B0", "#A0B0C0", "#90A4B8", "#B8C4D0",
    ],
    "莫兰迪粉棕 (Morandi Rose)": [
        "#C4A8A8", "#D4B8B0", "#C8B0A8", "#D0BCB4",
        "#BCA4A0", "#CCB4AC", "#C0ACA4", "#D8C4BC",
    ],
    "柔和森林 (Soft Forest)": [
        "#A8C4A8", "#B4D0B4", "#ACC8AC", "#C0D8C0",
        "#A0B8A0", "#B8CCB8", "#A4C0A4", "#C8DCC8",
    ],
    "暖灰调 (Warm Grey)": [
        "#B8A8A0", "#C4B8B0", "#BCB0A8", "#D0C4BC",
        "#B0A098", "#C8BCB4", "#B4A89C", "#D8CCC4",
    ],
    "冷蓝调 (Cool Blue)": [
        "#A0B8D0", "#B0C8E0", "#A8C0D8", "#BCD0E4",
        "#98B0C8", "#B4C4D8", "#A4B8CC", "#C4D8EC",
    ],
    "淡雅紫调 (Soft Lavender)": [
        "#C0B8D0", "#D0C8E0", "#C8C0D8", "#D8D0E8",
        "#B8B0C8", "#CCC4D8", "#C4B8D0", "#E0D8F0",
    ],
    "经典学术 (Classic Academic)": [
        "#8FB8DE", "#E8A8A8", "#A8D0A8", "#F0D890",
        "#C0A8D8", "#F0C0A0", "#90C8C8", "#D0B0B0",
    ],
}


def _empty_fig(message="No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(size=14, color="#999"))
    return fig


def prism_bar_chart(
    summary_df: pd.DataFrame,
    result_df: pd.DataFrame,
    stats_results: dict,
    target_gene: str,
    palette_name: str = "Cell 期刊 (Cell)",
    error_type: str = "SEM",
) -> go.Figure:
    """GraphPad Prism-style bar chart — 600x600, solid dots, sig brackets.

    Args:
        summary_df: From compute_summary().
        result_df: Per-sample results from compute().
        stats_results: From statistics.run_pipeline().
        target_gene: Gene name for title and filtering.
        palette_name: Key from PALETTES dict.
        error_type: "SEM" or "SD".
    """
    if summary_df.empty or "target_gene" not in summary_df.columns:
        return _empty_fig()

    available = summary_df["target_gene"].unique().tolist()
    if target_gene not in available:
        for a in available:
            if target_gene in a or a in target_gene:
                target_gene = a
                break

    sdf = summary_df[summary_df["target_gene"] == target_gene].copy()
    rdf = result_df[result_df["target_gene"] == target_gene].copy()
    if sdf.empty:
        return _empty_fig(f"No data for {target_gene}")

    groups = sdf["group"].tolist()
    n_groups = len(groups)
    colors = PALETTES.get(palette_name, PALETTES["Cell 期刊 (Cell)"])
    group_colors = [colors[i % len(colors)] for i in range(n_groups)]
    group_to_x = {g: i for i, g in enumerate(groups)}

    fc_means = sdf["fc_mean"].tolist()
    if error_type == "SD":
        fc_errors = sdf["fc_sd"].tolist()
    else:
        fc_errors = sdf["fc_sem"].tolist()

    fig = go.Figure()

    # ── Layer 1: Bars ──
    fig.add_trace(go.Bar(
        x=list(range(n_groups)),
        y=fc_means,
        marker=dict(
            color=group_colors,
            opacity=0.85,
            line=dict(color="black", width=0.6),
        ),
        name="Mean",
        showlegend=False,
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "Mean = %{y:.4f}<br>"
            f"{error_type} = %{{error_y.array:.4f}}"
            "<extra></extra>"
        ),
        customdata=groups,
        error_y=dict(
            type="data",
            array=fc_errors,
            visible=True,
            thickness=1.5,
            width=8,
            color="black",
        ),
    ))

    # ── Layer 2: Solid black dots (filled circles) ──
    rng = np.random.default_rng(42)
    for i, grp in enumerate(groups):
        grp_vals = rdf.loc[rdf["group"] == grp, "normalized_data"].dropna()
        if len(grp_vals) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, len(grp_vals))
        fig.add_trace(go.Scatter(
            x=np.full(len(grp_vals), float(i)) + jitter,
            y=grp_vals.values,
            mode="markers",
            marker=dict(
                color="black",
                symbol="circle",
                size=10,
                line=dict(width=0),
            ),
            name=f"{grp} (n={len(grp_vals)})",
            showlegend=False,
            hovertemplate="%{y:.4f}<extra></extra>",
        ))

    # ── Layer 3: Significance brackets ──
    brackets = get_significance_brackets(
        stats_results, groups, target_gene, group_to_x,
    )

    if brackets:
        all_y_vals = []
        for grp in groups:
            vals = rdf.loc[rdf["group"] == grp, "normalized_data"].dropna()
            all_y_vals.extend(vals.tolist())
        if all_y_vals:
            y_max = max(all_y_vals)
            y_range = (
                y_max - min(all_y_vals)
                if len(set(all_y_vals)) > 1
                else y_max * 0.1
            )
        else:
            y_max = max(fc_means) if fc_means else 1
            y_range = y_max * 0.2

        brackets.sort(key=lambda b: b["x1"] - b["x0"], reverse=True)

        for tier, b in enumerate(brackets):
            x0, x1 = b["x0"], b["x1"]
            y_bracket = y_max + y_range * 0.15 + tier * y_range * 0.18
            y_text = y_bracket + y_range * 0.04

            fig.add_shape(
                type="line",
                x0=x0, x1=x1, y0=y_bracket, y1=y_bracket,
                line=dict(color="black", width=1.2),
            )
            fig.add_shape(
                type="line",
                x0=x0, x1=x0,
                y0=y_bracket - y_range * 0.02, y1=y_bracket,
                line=dict(color="black", width=1.2),
            )
            fig.add_shape(
                type="line",
                x0=x1, x1=x1,
                y0=y_bracket - y_range * 0.02, y1=y_bracket,
                line=dict(color="black", width=1.2),
            )

            fig.add_annotation(
                x=(x0 + x1) / 2,
                y=y_text,
                text=b["label"],
                showarrow=False,
                font=dict(size=16, color="black", family="Arial"),
            )

    # ── Prism-style layout (800x800 square, journal-ready) ──
    FONT_SIZE = 15
    TICK_SIZE = 14
    TITLE_SIZE = 20

    # Auto-tilt X-axis labels if any group name is long
    max_label_len = max(len(str(g)) for g in groups) if groups else 0
    xaxis_kw = dict(
        tickvals=list(range(n_groups)),
        ticktext=groups,
        title="",
        showline=True,
        linecolor="black",
        linewidth=1.2,
        ticks="outside",
        tickcolor="black",
        tickwidth=1.2,
        ticklen=6,
        tickfont=dict(color="black", size=TICK_SIZE, family="Arial"),
        showgrid=False,
        zeroline=False,
    )
    if max_label_len > 6:
        xaxis_kw["tickangle"] = -30
        xaxis_kw["tickfont"] = dict(color="black", size=TICK_SIZE - 1, family="Arial")

    fig.update_xaxes(**xaxis_kw)

    fig.update_yaxes(
        title={
            "text": "Relative mRNA fold change",
            "font": {"size": FONT_SIZE, "color": "black", "family": "Arial"},
        },
        showline=True,
        linecolor="black",
        linewidth=1.2,
        ticks="outside",
        tickcolor="black",
        tickwidth=1.2,
        ticklen=6,
        tickfont=dict(color="black", size=TICK_SIZE, family="Arial"),
        showgrid=False,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1.0,
    )

    fig.update_layout(
        width=800,
        height=800,
        plot_bgcolor="white",
        paper_bgcolor="white",
        bargap=0.30,
        margin=dict(l=80, r=50, t=100, b=80),
        title={
            "text": target_gene,
            "font": {
                "size": TITLE_SIZE, "color": "black",
                "family": "Arial, sans-serif",
            },
            "x": 0.5,
            "xanchor": "center",
        },
        font=dict(family="Arial, sans-serif", size=FONT_SIZE, color="black"),
        showlegend=False,
        hovermode="closest",
    )

    return fig


def fig_to_bytes(
    fig: go.Figure,
    fmt: str = "png",
) -> bytes:
    """Export Plotly figure to png / svg / pdf bytes.

    All formats use 800x800 square canvas.
    PNG uses scale=3 for 300+ DPI journal submission quality.
    """
    if fmt == "svg":
        return fig.to_image(format="svg", width=800, height=800)
    elif fmt == "pdf":
        return fig.to_image(format="pdf", width=800, height=800)
    return fig.to_image(format="png", width=800, height=800, scale=3)
