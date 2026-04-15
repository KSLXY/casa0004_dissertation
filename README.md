# casa0004_dissertation

## 中文说明
本仓库包含 Casa0004 论文项目的可复现建模流水线与 Streamlit 可视化看板。

### 快速开始
```bash
pip install -r requirements.txt
streamlit run reproducible_pipeline/streamlit_app.py
```

### 仓库包含内容
- 可复现 pipeline 代码（聚类、邻近站点、事件特征、消融实验）
- Streamlit 可视化看板（支持中英文切换）
- 轻量级结果产物（指标、消融、重要性、报告与图表）

### 数据说明
仓库默认保留 `run_20260415_105036` 的轻量结果文件；超大文件（如完整 predictions/split）已排除，便于 GitHub 与 Streamlit Cloud 部署。

---

## English
This repository contains the reproducible modeling pipeline and Streamlit dashboard for the Casa0004 dissertation project.

### Quick Start
```bash
pip install -r requirements.txt
streamlit run reproducible_pipeline/streamlit_app.py
```

### What is included
- Reproducible pipeline code (clustering, neighbor features, event features, ablation)
- Streamlit dashboard (with Chinese/English language switch)
- Lightweight run artifacts (metrics, ablation outputs, feature importance, reports, and figures)

### Data notes
The repository includes lightweight outputs from `run_20260415_105036`. Very large files (e.g., full predictions/splits) are excluded for GitHub and Streamlit Cloud deployment.
