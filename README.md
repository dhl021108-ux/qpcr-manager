# 🧬 qPCR 数据管理器

> 对标 GraphPad Prism 规范的学术级 qPCR 自动化分析与可视化工具。支持交互式表格输入、严格 6 步法 $2^{-\Delta\Delta Ct}$ 计算、自动统计检验，一键导出 Nature / Cell 风格正方形矢量图表。

---

## ✨ 核心特性

- **📊 交互式数据网格** — 基于 `st.data_editor`，支持双击编辑、Delete 清空、从 Excel 直接 Ctrl+V 粘贴 Ct 值矩阵
- **🧪 严格 6 步法计算** — 全局对照组基准归一化，逐孔计算 $\Delta Ct$ → mean1 → $\Delta\Delta Ct$ → $2^{-\Delta\Delta Ct}$ → mean2 → mean3 → 归一化数据，一步不漏
- **📈 智能统计检验** — 基于 $\Delta Ct$ 对数尺度（对标 Prism），2 组自动 t-test，3+ 组自动 One-way ANOVA + 两两比较；三级显著性标注（\*, \*\*, \*\*\*）
- **🎨 10 套学术配色** — Cell / Nature / Science / 莫兰迪色系，柔和护眼，适合期刊发表
- **📉 Prism 风格可视化** — 纯白背景、黑色坐标轴、实心数据点、误差棒 (SEM/SD)、显著性对比括号 + 星号标注
- **📤 多格式导出** — 计算结果 Excel、统计检验 Excel、PNG (300+ DPI) / SVG / PDF 矢量图表，全部正方形 800×800 画板
- **🗂️ 合并单元格输出** — 结果表格自动合并 3+ 连续相同值，分组/样品/均值列视觉居中，学术记录规范

---

## 🚀 本地运行

### 环境要求

- Python >= 3.10

### 安装

```bash
# 1. 克隆或下载项目
cd qpcr-manager

# 2. (推荐) 创建虚拟环境
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

### 启动

```bash
streamlit run app.py
```

浏览器自动打开 **http://localhost:8501**。

---

## 📖 使用说明

### 数据输入

页面加载后默认展示 **3 组 × 3 生物学重复 × 3 技术重复 = 27 行** 的空白表格。

| 操作 | 方法 |
|------|------|
| 修改单元格 | 双击 → 输入 Ct 值 |
| 清空数据 | 选中单元格 → Delete / Backspace |
| 从 Excel 粘贴 | Excel 中 Ctrl+C → 点击表格起始格 → Ctrl+V |
| 调整组别 | 左侧「实验分组」下拉框添加/删除组 |
| 调整重复数 | 「每组的生物学重复数」控制 2–10 |
| 修改基因名 | 直接编辑「内参基因名称」/「目的基因名称」文本框 |

### 数据格式要求

- 每个生物学样本默认 **3 个技术重复孔**（3 行）
- Ct 值应为正数，缺失值留空
- 第一组默认作为**对照组**（校准样本）

### 分析流程

1. **配置参数** — 选择分组、生物学重复数、基因名称
2. **填入 Ct 值** — 编辑表格或从 Excel 粘贴
3. **选择检验方法** — 系统自动推荐（2 组→t-test，3+ 组→ANOVA），可手动更改
4. **点击「开始分析」** — 自动完成 6 步计算、统计检验、图表生成
5. **导出结果** — 下载 Excel 汇总表、统计检验结果、PNG/SVG/PDF 图表

---

## 🧮 计算方法

采用 **全局对照组基准的 6 步逐孔 $\Delta\Delta$Ct 法**：

| 步骤 | 公式 | 粒度 |
|------|------|------|
| 1 | $\Delta Ct = Ct_{target} - Ct_{ref}$；mean1 = AVG(同一样品 3 孔 $\Delta Ct$) | 每孔 |
| 2 | 全局基准 = AVG(所有 CONTROL 样品 mean1) | 全板 |
| 3 | $\Delta\Delta Ct = \Delta Ct$ − 全局基准 | 每孔 |
| 4 | $2^{-\Delta\Delta Ct}$；mean2 = AVG(同一样品 3 孔 $2^{-\Delta\Delta Ct}$) | 每孔 |
| 5 | mean3 = AVG(同组所有样品 mean2) | 每组 |
| 6 | 归一化数据 = mean2 / CONTROL 组 mean3 | 每样品 |

统计检验基于 $\Delta Ct$ 对数尺度（近似正态分布），不使用 $2^{-\Delta\Delta Ct}$ 计算 P 值。

---

## 📁 项目结构

```
qpcr-manager/
├── app.py                  # Streamlit 主应用
├── requirements.txt        # Python 依赖
├── README.md               # 项目说明
├── Dockerfile              # Docker 部署
├── src/
│   ├── parser.py           # Excel/CSV 数据解析
│   ├── calculator.py       # 6 步法 ΔΔCt 计算引擎
│   ├── statistics.py       # t-test / ANOVA 统计检验
│   ├── visualizer.py       # Prism 风格 Plotly 图表
│   └── exporter.py         # Excel / PDF 导出
├── templates/              # Excel 模板
├── examples/               # 示例数据
└── tests/                  # 单元测试
```

---

## 🧪 运行测试

```bash
pip install pytest
pytest tests/ -v
```

---

## 📚 参考文献

- Livak KJ, Schmittgen TD. Analysis of relative gene expression data using real-time quantitative PCR and the $2^{-\Delta\Delta Ct}$ method. *Methods*. 2001;25(4):402–408.
- GraphPad Prism 10 Statistics Guide. GraphPad Software, 2024.

---

## 📄 许可证

MIT License
