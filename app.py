# -*- coding: utf-8 -*-
"""qPCR 数据管理器 — 交互式数据网格 + 6步逐孔 ΔΔCt 分析 + 可视化

使用方法: streamlit run app.py
"""

import io
import json
import re
from io import BytesIO
from datetime import datetime
from hashlib import sha256

import numpy as np
import pandas as pd
import streamlit as st

from src.calculator import compute_full_table, compute_summary
from src.exporter import plot_data_to_tsv
from src.statistics import run_pipeline, stats_to_dataframe
from src.visualizer import prism_bar_chart, fig_to_bytes, PALETTES
from src.tracker import track_login, get_user_stats

# Dynamic date prefix for export filenames
TODAY = datetime.now().strftime("%Y.%m.%d")

# ── 页面设置 ──────────────────────────────────────────────────
st.set_page_config(
    page_title="qPCR 数据管理器",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 常量 ─────────────────────────────────────────────────────
DEFAULT_GROUPS = ["对照组", "实验组1", "实验组2"]
DEFAULT_TECH_REPS = 3
MAX_TECH_REPS = 6
MAX_BIO_REPS = 10
MIN_BIO_REPS = 2
DEFAULT_BIO_REPS = 3

EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

# ── 缓存计算函数（高并发优化） ───────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def cached_compute_full_table(_df_json, ref_col, target_col,
                               control_group, sample_col, group_col):
    """Cached wrapper for the 6-step pipeline."""
    df = pd.read_json(io.StringIO(_df_json))
    return compute_full_table(df, ref_col, target_col, control_group,
                              sample_col, group_col)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_compute_summary(_result_json):
    """Cached wrapper for summary aggregation."""
    result_df = pd.read_json(io.StringIO(_result_json))
    return compute_summary(result_df)


@st.cache_data(show_spinner=False, ttl=3600)
def cached_run_stats(_result_json, control_group, test_method):
    """Cached wrapper for statistical tests."""
    result_df = pd.read_json(io.StringIO(_result_json))
    return run_pipeline(result_df, control_group, test_method=test_method)


def cached_prism_chart(_summary_json, _result_json, _stats_json,
                       target_gene, palette_name, error_type):
    """Wrapper for Plotly chart generation — NO cache, must re-render every time."""
    summary_df = pd.read_json(io.StringIO(_summary_json))
    result_df = pd.read_json(io.StringIO(_result_json))
    stats_results = json.loads(_stats_json)
    return prism_bar_chart(summary_df, result_df, stats_results,
                           target_gene, palette_name, error_type)


# ── 会话状态初始化 ───────────────────────────────────────────
defaults = {
    "groups": DEFAULT_GROUPS.copy(),
    "tech_reps": DEFAULT_TECH_REPS,
    "bio_reps_map": {g: DEFAULT_BIO_REPS for g in DEFAULT_GROUPS},
    "ref_gene_name": "GAPDH",
    "target_gene_name": "IL-6",
    "editor_df": None,
    "result_df": None,
    "summary_df": None,
    "stats_results": None,
    "full_table": None,
    "computed": False,
    "last_groups_hash": "",
    "last_tech_hash": "",
    "last_bio_hash": "",
    "last_ref_name": "",
    "last_target_name": "",
    # Login state
    "logged_in": False,
    "user_email": None,
    "login_shown": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── 免密邮箱登录弹窗 ─────────────────────────────────────────

def show_login():
    """Show the one-time login dialog."""
    st.markdown("## 欢迎使用 qPCR 数据管理器")
    st.caption("一款便捷的 qPCR 数据处理工具")

    email = st.text_input(
        "📧 邮箱地址（选填，无需密码）",
        placeholder="your@institution.edu",
        key="login_email_input",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("✅ 验证并登录", use_container_width=True, type="primary"):
            if not email.strip():
                st.error("请输入邮箱地址")
            elif not EMAIL_RE.match(email.strip()):
                st.error("邮箱格式不正确，请重新输入")
            else:
                email_clean = email.strip().lower()
                result = track_login(email_clean)
                st.session_state.user_email = email_clean
                st.session_state.logged_in = True
                st.session_state.login_shown = True
                st.success(
                    f"登录成功！欢迎回来，这是您第 **{result['usage_count']}** 次使用。"
                )
                st.rerun()

    with c2:
        if st.button("⏭ 跳过，直接使用", use_container_width=True):
            st.session_state.user_email = None
            st.session_state.logged_in = True
            st.session_state.login_shown = True
            st.rerun()

    st.divider()
    st.caption("跳过登录不会影响任何功能，登录仅用于记录使用次数。")

    # Show welcome-back message for returning users
    if st.session_state.user_email:
        stats = get_user_stats(st.session_state.user_email)
        if stats:
            st.info(
                f"欢迎回来 **{st.session_state.user_email}** ！"
                f"上次使用：{stats['last_login_date']}，"
                f"累计使用 **{stats['usage_count']}** 次。"
            )


if not st.session_state.logged_in:
    show_login()
    st.stop()


# ── 工具函数 ─────────────────────────────────────────────────

def build_default_df(groups, bio_reps_map, tech_reps, ref_col="GAPDH Ct", target_col="IL-6 Ct"):
    """构建初始数据表格 — 支持每组独立的生物学重复数和可变技术重复数。"""
    rows = []
    sample_num = 1
    for grp in groups:
        n_bio = bio_reps_map.get(grp, DEFAULT_BIO_REPS)
        for bio in range(1, n_bio + 1):
            for tech in range(1, tech_reps + 1):
                rows.append({
                    "样本": f"样本 {sample_num}",
                    "分组": grp,
                    ref_col: np.nan,
                    target_col: np.nan,
                })
            sample_num += 1
    return pd.DataFrame(rows)


def rebuild_df_preserving(old_df, groups, bio_reps_map, tech_reps, ref_col, target_col):
    """重建数据结构时保留已输入的 Ct 值。"""
    new_df = build_default_df(groups, bio_reps_map, tech_reps, ref_col, target_col)
    if old_df is not None and not old_df.empty:
        old_sample_col = "样本" if "样本" in old_df.columns else "Sample"
        old_group_col = "分组" if "分组" in old_df.columns else "Group"
        for col in [ref_col, target_col]:
            if col in old_df.columns and col in new_df.columns:
                for idx, row in new_df.iterrows():
                    match = old_df.loc[
                        (old_df[old_sample_col] == row["样本"]) &
                        (old_df[old_group_col] == row["分组"])
                    ]
                    if not match.empty and col in match.columns:
                        val = match[col].values[0]
                        if pd.notna(val):
                            new_df.at[idx, col] = val
    return new_df


def build_result_html(full_table, ref_name, target_name):
    """构建逐孔结果表格 (HTML)，智能合并且格，保留原始行顺序。

    3 个或以上连续相同值的单元格自动合并 (rowspan)，垂直居中。
    """
    if full_table is None or full_table.empty:
        return "<p>暂无结果数据</p>"

    # Preserve original row order — NO sorting
    df = full_table.copy().reset_index(drop=True)
    n_rows = len(df)

    # Columns to display (in order)
    display_cols = [
        ("分组",      "分组",            "center"),
        ("样本",      "样本",            "center"),
        ("ct_target", f"{target_name} Ct", "right"),
        ("ct_ref",    f"{ref_name} Ct",    "right"),
        ("delta_ct",       "ΔCt",        "right"),
        ("mean1",          "mean1",       "right"),
        ("delta_delta_ct", "ΔΔCt",       "right"),
        ("fc_per_well",    "2^(-ΔΔCt)",  "right"),
        ("mean2",          "mean2",       "right"),
        ("mean3",          "mean3",       "right"),
        ("归一化数据",     "归一化数据",  "right"),
        ("_well_selected", "筛选",         "center"),
    ]

    fmt = {
        "ct_target": "{:.2f}", "ct_ref": "{:.2f}",
        "delta_ct": "{:.2f}", "mean1": "{:.2f}",
        "delta_delta_ct": "{:.3f}", "fc_per_well": "{:.4f}",
        "mean2": "{:.4f}", "mean3": "{:.4f}", "归一化数据": "{:.4f}",
    }

    # ── Precompute rowspan metadata for each column ─────────
    # For each cell: (should_render: bool, rowspan: int)
    col_meta = {}
    for key, _, _ in display_cols:
        meta = []
        i = 0
        while i < n_rows:
            val = df.iloc[i].get(key, np.nan)
            j = i + 1
            while j < n_rows:
                next_val = df.iloc[j].get(key, np.nan)
                if pd.isna(val) and pd.isna(next_val):
                    j += 1
                elif (not pd.isna(val)) and (not pd.isna(next_val)) and val == next_val:
                    j += 1
                else:
                    break
            run_len = j - i
            if run_len >= 3:
                meta.append((True, run_len))
                for _ in range(1, run_len):
                    meta.append((False, 0))
            else:
                for _ in range(run_len):
                    meta.append((True, 1))
            i = j
        col_meta[key] = meta

    # ── Build HTML ──────────────────────────────────────────
    css = '''
    <style>
    .qpcr-table { border-collapse: collapse; width: 100%; font-size: 13px;
                  font-family: Arial, sans-serif; }
    .qpcr-table th { background: #4472C4; color: white; padding: 10px 12px;
                     border: 1px solid #3a5fa8; text-align: center;
                     font-weight: 600; }
    .qpcr-table td { padding: 6px 10px; border: 1px solid #ddd; }
    .qpcr-table .merged { vertical-align: middle; text-align: center;
                          background: #f5f7fa; font-weight: 500; }
    .qpcr-table .num-r { text-align: right; font-variant-numeric: tabular-nums; }
    .qpcr-table .num-c { text-align: center; font-variant-numeric: tabular-nums;
                         vertical-align: middle; }
    .qpcr-table tr:nth-child(even) td { background: #fafbfc; }
    .qpcr-table tr:nth-child(even) td.merged { background: #eef1f5; }
    .qpcr-table tr.rejected { color: #999; background: #fff5f5; }
    .qpcr-table tr.rejected td { background: #fff5f5; }
    .qpcr-table td.rejected-mark { color: #cc5555; font-weight: bold; }
    </style>
    '''

    html = css + '<table class="qpcr-table"><thead><tr>'
    for _, label, _ in display_cols:
        html += f'<th>{label}</th>'
    html += '</tr></thead><tbody>'

    for row_idx in range(n_rows):
        is_selected = df.iloc[row_idx].get("_well_selected", True)
        row_class = ' class="rejected"' if is_selected == False else ''
        html += f'<tr{row_class}>'
        for key, _, align in display_cols:
            render, rowspan = col_meta[key][row_idx]
            if not render:
                continue  # consumed by a previous rowspan

            val = df.iloc[row_idx].get(key, np.nan)
            if key == "_well_selected":
                text = "✓" if val == True else ("✗" if val == False else "—")
            elif key in fmt and pd.notna(val):
                text = fmt[key].format(val)
            elif key in fmt:
                text = "N/A"
            else:
                text = str(val) if pd.notna(val) else "N/A"

            if rowspan > 1:
                if key == "_well_selected" and val == False:
                    cls = "merged rejected-mark"
                else:
                    cls = "merged num-c" if align == "right" else "merged"
                html += (
                    f'<td class="{cls}" rowspan="{rowspan}"'
                    f' style="vertical-align: middle; text-align: center;">'
                    f'{text}</td>'
                )
            else:
                if key == "_well_selected" and val == False:
                    cls = "rejected-mark"
                else:
                    cls = "num-r" if align == "right" else ""
                html += f'<td class="{cls}">{text}</td>'

        html += '</tr>'

    html += '</tbody></table>'
    return html


# ── 标题 ─────────────────────────────────────────────────────
st.title("🧬 qPCR 数据管理器")
st.caption("交互式数据网格  ·  6步逐孔 ΔΔCt 分析  ·  统计检验  ·  可视化")

# ── 参数配置区 ───────────────────────────────────────────────
st.header("⚙️ 实验参数配置")

ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

with ctrl_col1:
    new_groups = st.multiselect(
        "实验分组",
        options=["对照组", "实验组1", "实验组2", "实验组3", "实验组4", "实验组5",
                 "处理组", "WT", "KO", "OE", "KD", "A组", "B组", "C组", "D组"],
        default=st.session_state.groups,
        help="添加或删除实验组。第一个组将作为对照组（校准样本）。",
    )
    if not new_groups:
        new_groups = ["对照组"]
    if set(new_groups) != set(st.session_state.groups):
        st.session_state.groups = new_groups
        # Sync bio_reps_map: retain existing counts, add defaults for new groups
        old_map = st.session_state.bio_reps_map
        new_map = {}
        for g in new_groups:
            new_map[g] = old_map.get(g, DEFAULT_BIO_REPS)
        st.session_state.bio_reps_map = new_map
        st.rerun()

with ctrl_col2:
    tech_reps = st.selectbox(
        "每样本技术重复孔数",
        options=list(range(3, MAX_TECH_REPS + 1)),
        index=st.session_state.tech_reps - 3,
        help="每个生物学样本设置几个技术重复孔（3–6）。超过 3 孔时自动智能筛选最优 3 孔。",
    )
    if tech_reps != st.session_state.tech_reps:
        st.session_state.tech_reps = tech_reps
        st.rerun()

with ctrl_col3:
    ref_gene_name = st.text_input(
        "内参基因名称",
        value=st.session_state.ref_gene_name,
        help="例如：GAPDH、ACTB、18S rRNA",
    )
    if ref_gene_name != st.session_state.ref_gene_name:
        st.session_state.ref_gene_name = ref_gene_name
        st.rerun()

with ctrl_col4:
    target_gene_name = st.text_input(
        "目的基因名称",
        value=st.session_state.target_gene_name,
        help="例如：IL-6、TNF-α、TP53",
    )
    if target_gene_name != st.session_state.target_gene_name:
        st.session_state.target_gene_name = target_gene_name
        st.rerun()

# ── Per-group biological replicate count ──
if len(st.session_state.groups) > 0:
    st.caption("每组生物学重复数（可不等）：")
    bio_cols = st.columns(min(len(st.session_state.groups), 5))
    bio_map_changed = False
    for i, grp in enumerate(st.session_state.groups):
        col_idx = i % 5
        with bio_cols[col_idx]:
            new_val = st.number_input(
                f"{grp}",
                min_value=MIN_BIO_REPS,
                max_value=MAX_BIO_REPS,
                value=st.session_state.bio_reps_map.get(grp, DEFAULT_BIO_REPS),
                step=1,
                key=f"bio_rep_{grp}",
            )
            if new_val != st.session_state.bio_reps_map.get(grp, DEFAULT_BIO_REPS):
                st.session_state.bio_reps_map[grp] = new_val
                bio_map_changed = True
    if bio_map_changed:
        st.rerun()

# ── 列名 ─────────────────────────────────────────────────────
ref_ct_col = f"{ref_gene_name} Ct"
target_ct_col = f"{target_gene_name} Ct"

# ── 维护编辑器 DataFrame ─────────────────────────────────────
groups_hash = "|".join(st.session_state.groups)
bio_hash = json.dumps(st.session_state.bio_reps_map, sort_keys=True)
tech_hash = str(st.session_state.tech_reps)

need_rebuild = (
    groups_hash != st.session_state.last_groups_hash or
    bio_hash != st.session_state.last_bio_hash or
    tech_hash != st.session_state.last_tech_hash
)
need_rename = (
    st.session_state.ref_gene_name != st.session_state.last_ref_name or
    st.session_state.target_gene_name != st.session_state.last_target_name
)

if need_rebuild or need_rename:
    new_df = rebuild_df_preserving(
        st.session_state.editor_df,
        st.session_state.groups,
        st.session_state.bio_reps_map,
        st.session_state.tech_reps,
        ref_ct_col,
        target_ct_col,
    )
    st.session_state.editor_df = new_df
    st.session_state.computed = False

if st.session_state.editor_df is None:
    st.session_state.editor_df = build_default_df(
        st.session_state.groups,
        st.session_state.bio_reps_map,
        st.session_state.tech_reps,
        ref_ct_col, target_ct_col,
    )

st.session_state.last_groups_hash = groups_hash
st.session_state.last_bio_hash = bio_hash
st.session_state.last_tech_hash = tech_hash
st.session_state.last_ref_name = ref_gene_name
st.session_state.last_target_name = target_gene_name

# ── 数据编辑器 ───────────────────────────────────────────────
st.divider()
st.header("📊 数据输入")

st.info(
    "💡 **提示**：双击单元格可修改内容；选中单元格后按键盘 **Delete / Backspace** 键可快速清除数据；"
    "支持直接从 **Excel 复制粘贴**（Ctrl+C → 选中起始单元格 → Ctrl+V）。"
)

st.caption(
    f"每个生物学样本自动生成 {st.session_state.tech_reps} 行（{st.session_state.tech_reps} 个技术重复孔）。"
    f"{'超过 3 孔时自动智能筛选最优 3 孔。' if st.session_state.tech_reps > 3 else ''}"
)

editor_df = st.session_state.editor_df.copy()
data_cols = [c for c in editor_df.columns
             if c not in ("2^(-ΔΔct)", "归一化数据")]

col_config = {
    "样本": st.column_config.TextColumn("样本", help="样本标识", width="medium"),
    "分组": st.column_config.SelectboxColumn(
        "分组", options=st.session_state.groups, help="分组标签", width="medium",
    ),
}
if "Sample" in editor_df.columns:
    col_config["Sample"] = st.column_config.TextColumn("Sample", width="medium")
if "Group" in editor_df.columns:
    col_config["Group"] = st.column_config.SelectboxColumn(
        "Group", options=st.session_state.groups, width="medium",
    )

for c in data_cols:
    if c not in ("样本", "分组", "Sample", "Group"):
        col_config[c] = st.column_config.NumberColumn(
            c, help=f"{c} 的 Ct 值", format="%.2f", min_value=0.0, max_value=45.0,
        )

if "2^(-ΔΔct)" in editor_df.columns:
    col_config["2^(-ΔΔct)"] = st.column_config.NumberColumn(
        "2^(-ΔΔct)", format="%.4f", disabled=True,
    )
if "归一化数据" in editor_df.columns:
    col_config["归一化数据"] = st.column_config.NumberColumn(
        "归一化数据", format="%.4f", disabled=True,
    )

display_df = editor_df[data_cols]
if st.session_state.computed:
    for rc in ["2^(-ΔΔct)", "归一化数据"]:
        if rc in editor_df.columns:
            display_df[rc] = editor_df[rc]

edited = st.data_editor(
    display_df,
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic",
    column_config=col_config,
    key="qpcr_editor",
)

st.session_state.editor_df = edited

# ── 计算按钮 + 统计检验方法选择 ─────────────────────────────
st.divider()
st.header("🔬 分析与检验")

control_group = st.session_state.groups[0] if st.session_state.groups else "对照组"

n_effective_groups = len(st.session_state.groups)
if n_effective_groups >= 3:
    recommended_test = "anova"
    recommended_label = "单因素方差分析 (One-way ANOVA)"
elif n_effective_groups == 2:
    recommended_test = "ttest"
    recommended_label = "Student's t 检验"
else:
    recommended_test = "none"
    recommended_label = "不进行检验（分组不足）"

calc_col1, calc_col2, calc_col3 = st.columns([1, 1, 1])
with calc_col1:
    do_calc = st.button("🧪 开始分析", type="primary", use_container_width=True)

with calc_col2:
    test_options = {
        "auto": "智能自动选择",
        "ttest": "Student's t 检验",
        "mannwhitney": "Mann-Whitney U 检验",
        "anova": "单因素方差分析 (ANOVA)",
        "none": "不进行统计检验",
    }
    test_keys = list(test_options.keys())
    default_idx = test_keys.index(recommended_test) if recommended_test in test_keys else 0

    selected_test = st.selectbox(
        "统计检验方法",
        options=test_keys,
        index=default_idx,
        format_func=lambda x: f"{test_options[x]} {'(推荐)' if x == recommended_test else ''}",
        help=f"系统推荐：{recommended_label}。您可手动更改。",
    )

with calc_col3:
    st.caption(f"对照组：**{control_group}**")
    st.caption(f"有效分组数：**{n_effective_groups}** 组")

if do_calc:
    work_df = edited.copy()
    sample_col = "样本" if "样本" in work_df.columns else "Sample"
    group_col = "分组" if "分组" in work_df.columns else "Group"

    if ref_ct_col in work_df.columns and target_ct_col in work_df.columns:
        work_df = work_df.dropna(subset=[ref_ct_col, target_ct_col], how="all")
    work_df[group_col] = work_df[group_col].ffill()

    try:
        # Use cached wrappers for 100-user concurrency
        full_table, per_sample_result = cached_compute_full_table(
            work_df.to_json(),
            ref_ct_col, target_ct_col, control_group,
            sample_col, group_col,
        )
        summary_df = cached_compute_summary(per_sample_result.to_json())

        if selected_test == "none":
            stats_results = {}
        else:
            pipeline_test = selected_test if selected_test in ("ttest", "mannwhitney") else "auto"
            stats_results = cached_run_stats(
                per_sample_result.to_json(), control_group, pipeline_test,
            )

        st.session_state.full_table = full_table
        st.session_state.result_df = per_sample_result
        st.session_state.summary_df = summary_df
        st.session_state.stats_results = stats_results
        st.session_state.computed = True
        st.rerun()
    except Exception as e:
        st.error(f"计算出错：{e}")
        import traceback
        with st.expander("调试详情"):
            st.code(traceback.format_exc())

# ── 结果展示 ─────────────────────────────────────────────────
if st.session_state.computed:
    result_df = st.session_state.result_df
    summary_df = st.session_state.summary_df
    stats_results = st.session_state.stats_results
    full_table = st.session_state.full_table

    st.divider()
    st.header("📈 分析结果")

    # ── 逐孔详细结果表格 (HTML 合并单元格) ──
    st.subheader("📋 逐孔计算结果")
    st.caption("按分组合并单元格，展示 6 步计算全流程的中间列。")

    if full_table is not None and not full_table.empty:
        html_table = build_result_html(full_table, ref_gene_name, target_gene_name)
        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.warning("无计算结果。")

    # ── 数据导出按钮（紧跟归一化结果表格之后）──
    exp_col1, _ = st.columns([1, 3])
    with exp_col1:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            if full_table is not None and not full_table.empty:
                full_table.to_excel(writer, sheet_name="逐孔计算结果", index=False)
            summary_df.to_excel(writer, sheet_name="分组汇总", index=False)
        st.download_button(
            "下载完整结果 (Excel)",
            data=output.getvalue(),
            file_name=f"{TODAY}_qPCR_{target_gene_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # ── 分组汇总 ──
    with st.expander("📊 分组汇总统计"):
        st.dataframe(
            summary_df.style.format({
                "fc_mean": "{:.4f}", "fc_sd": "{:.4f}", "fc_sem": "{:.4f}",
                "delta_ct_mean": "{:.2f}", "delta_ct_sem": "{:.2f}",
                "delta_delta_ct_mean": "{:.3f}", "delta_delta_ct_sem": "{:.3f}",
            }, na_rep="N/A"),
            use_container_width=True, hide_index=True,
        )

    # ── 统计检验 ──
    st.subheader("🔢 统计检验")
    if selected_test == "none" or not stats_results:
        st.info("已跳过统计检验（用户选择：不进行检验）。")
    else:
        st.caption("统计检验基于 ΔΔCt 值（对数尺度），对标 GraphPad Prism。")

        stats_df = stats_to_dataframe(stats_results)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        stats_excel = BytesIO()
        with pd.ExcelWriter(stats_excel, engine="openpyxl") as w:
            stats_df.to_excel(w, sheet_name="统计检验结果", index=False)
        st.download_button(
            "下载统计检验结果 (Excel)",
            data=stats_excel.getvalue(),
            file_name=f"{TODAY}_qPCR_{target_gene_name}_统计检验.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # ── 图表 ──
    st.divider()
    st.subheader("📉 GraphPad Prism 风格图表")

    palette_names = list(PALETTES.keys())
    col_pal, col_err = st.columns([1, 1])
    with col_pal:
        selected_palette = st.selectbox(
            "配色方案", options=palette_names, index=0,
        )
    with col_err:
        error_type = st.selectbox(
            "误差棒类型", options=["SEM", "SD"], index=0,
        )

    try:
        fig_prism = cached_prism_chart(
            summary_df.to_json(), result_df.to_json(),
            json.dumps(stats_results, default=str),
            target_ct_col, selected_palette, error_type,
        )
        st.plotly_chart(fig_prism, use_container_width=True, config={
            'displayModeBar': True,
            'toImageButtonOptions': {
                'format': 'svg',
                'filename': f'qPCR_{target_gene_name}',
                'scale': 1,
            },
        })
    except Exception as e:
        st.error(f"图表渲染出错：{e}")
        import traceback
        with st.expander("调试详情"):
            st.code(traceback.format_exc())
        fig_prism = None

    # ── 图表下载（Kaleido 0.1.0.post1，Streamlit Cloud Linux 完美运行）──
    if fig_prism is not None:
        png_data = fig_to_bytes(fig_prism, "png")
        pdf_data = fig_to_bytes(fig_prism, "pdf")
        tsv_str = plot_data_to_tsv(
            summary_df, result_df, stats_results,
            target_ct_col, error_type,
        )

        has_images = png_data is not None and pdf_data is not None
        if has_images:
            dl1, dl2, dl3 = st.columns(3)
            with dl1:
                st.download_button(
                    "下载 PNG (300dpi)", data=png_data,
                    file_name=f"{TODAY}_qPCR_{target_gene_name}.png",
                    mime="image/png",
                )
            with dl2:
                st.download_button(
                    "下载 PDF (矢量)", data=pdf_data,
                    file_name=f"{TODAY}_qPCR_{target_gene_name}.pdf",
                    mime="application/pdf",
                )
            with dl3:
                st.download_button(
                    "下载 TSV (作图数据)", data=tsv_str,
                    file_name=f"{TODAY}_qPCR_{target_gene_name}_plot_data.tsv",
                    mime="text/tab-separated-values",
                )
        else:
            st.download_button(
                "下载 TSV (作图数据)", data=tsv_str,
                file_name=f"{TODAY}_qPCR_{target_gene_name}_plot_data.tsv",
                mime="text/tab-separated-values",
            )

# ── 页脚 ─────────────────────────────────────────────────────
st.divider()
st.caption(
    "采用 6步逐孔 ΔΔCt 方法  ·  "
    "统计检验基于对数尺度  ·  "
    "所有数据均在本地处理，不上传任何服务器"
)
