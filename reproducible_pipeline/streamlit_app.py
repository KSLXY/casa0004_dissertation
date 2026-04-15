from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_BASELINE = "run_20260415_102043"

I18N = {
    "zh": {
        "title": "Casa0004 Pipeline 可视化看板",
        "lang": "语言 / Language",
        "run_select": "运行选择",
        "current_run": "当前 Run",
        "baseline_run": "基线 Run",
        "same_run": "当前 run 与基线 run 相同，可切换以查看提升对比。",
        "note": "说明",
        "note1": "旧版 run 缺少 ablation/spatial 文件时会自动降级显示。",
        "note2": "主比较指标：R²。",
        "tabs": ["概览", "流程工件", "结果消融", "特征解释", "预测误差", "文字结论"],
        "overview": "项目概览",
        "overview_desc": "该看板展示可复现建模流程：数据接入 -> 特征工程 -> 空间增强（聚类+邻居）-> 事件增强 -> 消融实验 -> 结果解释。",
        "overview_desc2": "本页面用于对比增强 run 与历史基线 run 的表现差异（重点看 R²）。",
        "feature_rows": "特征行数",
        "pred_rows": "预测行数",
        "summary_json": "当前 run 摘要（run_summary.json）",
        "baseline_label": "基线 Run",
        "pipeline": "流程与工件",
        "pipeline_info": "模块解释：每个模块均展示它的输入、核心处理逻辑、输出工件和作用。",
        "pipeline_steps": [
            "data_ingest: 校验输入 schema，生成 data_version.json",
            "cluster_stage: 自动搜索最佳 k，输出 cluster_artifacts/*",
            "neighbor_stage: 构建 station_neighbors.csv，生成邻居滞后特征",
            "event_stage: 合并节假日/学校假期/大型活动特征",
            "train: weekday/weekend x 5个scenario x 3个模型",
            "evaluate/report: 输出 metrics、ablation、图表与结论草稿",
        ],
        "cluster_summary": "聚类摘要",
        "neighbor_top": "邻接关系（前20行）",
        "missing_cluster": "当前 run 未找到 cluster_summary.csv",
        "missing_neighbor": "当前 run 未找到 station_neighbors.csv",
        "result": "结果与消融",
        "result_info": "模块解释：对比不同场景（baseline/cluster/neighbor/events/full）与不同模型，分析贡献来源。",
        "metrics_table": "完整指标表（可筛选）",
        "ablation_best": "Ablation 最优结果（每个 scenario x segment）",
        "vs_baseline": "与基线对照（最佳模型）",
        "feature": "特征解释",
        "feature_info": "模块解释：聚焦空间与事件特征的重要性，理解模型提升来自哪里。",
        "spatial_fi": "空间/事件特征重要性（spatial_feature_importance.csv）",
        "full_fi": "查看完整 feature_importance.csv",
        "pred": "预测与误差",
        "pred_info": "模块解释：查看预测样本、残差分布和站点级误差，定位模型强弱场景。",
        "resid_img": "残差分布图片（pipeline 自动生成）",
        "model_img": "模型对比图片（pipeline 自动生成）",
        "view_pred": "按场景/分段/模型查看预测样本",
        "err_station": "误差汇总（站点粒度）",
        "explain": "文字解释与结论",
        "explain_info": "模块解释：汇总实验结论，形成可用于论文/汇报的文本证据。",
        "conclusion": "结论草稿（conclusion_draft.md）",
        "report": "报告摘要（report.md）",
        "missing_generic": "当前 run 缺少该文件，已自动降级。",
    },
    "en": {
        "title": "Casa0004 Pipeline Dashboard",
        "lang": "Language / 语言",
        "run_select": "Run Selection",
        "current_run": "Current Run",
        "baseline_run": "Baseline Run",
        "same_run": "Current run equals baseline run; switch one run to see performance deltas.",
        "note": "Notes",
        "note1": "Legacy runs without ablation/spatial files are gracefully downgraded.",
        "note2": "Primary comparison metric: R².",
        "tabs": ["Overview", "Pipeline", "Results & Ablation", "Feature Insights", "Predictions & Errors", "Text & Conclusion"],
        "overview": "Overview",
        "overview_desc": "This dashboard visualizes the reproducible modeling pipeline: ingest -> features -> spatial enhancement (cluster + neighbors) -> event enhancement -> ablation -> interpretation.",
        "overview_desc2": "This page compares the enhanced run against the historical baseline run, mainly by R².",
        "feature_rows": "Feature Rows",
        "pred_rows": "Prediction Rows",
        "summary_json": "Run Summary (run_summary.json)",
        "baseline_label": "Baseline Run",
        "pipeline": "Pipeline & Artifacts",
        "pipeline_info": "Module explanation: each module shows its input, core logic, output artifacts, and purpose.",
        "pipeline_steps": [
            "data_ingest: validate input schema and generate data_version.json",
            "cluster_stage: search best k and output cluster_artifacts/*",
            "neighbor_stage: build station_neighbors.csv and neighbor lag features",
            "event_stage: merge public holidays, school breaks, and major events",
            "train: weekday/weekend x 5 scenarios x 3 models",
            "evaluate/report: export metrics, ablation, figures, and conclusion draft",
        ],
        "cluster_summary": "Cluster Summary",
        "neighbor_top": "Neighbor Links (top 20 rows)",
        "missing_cluster": "cluster_summary.csv is missing for this run.",
        "missing_neighbor": "station_neighbors.csv is missing for this run.",
        "result": "Results & Ablation",
        "result_info": "Module explanation: compare scenario/model combinations to identify incremental gains.",
        "metrics_table": "Full Metrics Table (filterable)",
        "ablation_best": "Best Ablation Result (per scenario x segment)",
        "vs_baseline": "Best-Model Comparison vs Baseline",
        "feature": "Feature Insights",
        "feature_info": "Module explanation: focus on spatial/event feature importance to explain performance gains.",
        "spatial_fi": "Spatial/Event Feature Importance (spatial_feature_importance.csv)",
        "full_fi": "Show full feature_importance.csv",
        "pred": "Predictions & Errors",
        "pred_info": "Module explanation: inspect samples, residual patterns, and station-level errors.",
        "resid_img": "Residual Distribution Figure (auto-generated)",
        "model_img": "Model Comparison Figure (auto-generated)",
        "view_pred": "View prediction samples by scenario/segment/model",
        "err_station": "Error Summary (station level)",
        "explain": "Text & Conclusion",
        "explain_info": "Module explanation: summarize evidence and narrative for dissertation/reporting use.",
        "conclusion": "Conclusion Draft (conclusion_draft.md)",
        "report": "Report Snapshot (report.md)",
        "missing_generic": "File missing for this run; fallback view is applied.",
    },
}


@st.cache_data(show_spinner=False)
def list_runs() -> List[Path]:
    return sorted([p for p in OUTPUTS_DIR.glob("run_*") if p.is_dir()])


@st.cache_data(show_spinner=False)
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_json_safe(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_text_safe(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")



def load_run_artifacts(run_dir: Path) -> Dict[str, object]:
    return {
        "metrics": read_csv_safe(run_dir / "metrics_all.csv"),
        "ablation": read_csv_safe(run_dir / "ablation_metrics.csv"),
        "feature_importance": read_csv_safe(run_dir / "feature_importance.csv"),
        "spatial_fi": read_csv_safe(run_dir / "spatial_feature_importance.csv"),
        "error_analysis": read_csv_safe(run_dir / "error_analysis.csv"),
        "predictions": read_csv_safe(run_dir / "predictions.csv"),
        "neighbors": read_csv_safe(run_dir / "station_neighbors.csv"),
        "cluster_summary": read_csv_safe(run_dir / "cluster_artifacts" / "cluster_summary.csv"),
        "summary": read_json_safe(run_dir / "run_summary.json"),
        "report": read_text_safe(run_dir / "report.md"),
        "conclusion": read_text_safe(run_dir / "conclusion_draft.md"),
        "model_comparison_img": run_dir / "model_comparison.png",
        "residual_img": run_dir / "residual_distribution.png",
    }



def ensure_scenario(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "scenario" not in df.columns:
        out = df.copy()
        out["scenario"] = "legacy"
        return out
    return df



def show_overview(data: Dict[str, object], current_run: Path, baseline_run: Path, t: Dict[str, object]) -> None:
    st.subheader(t["overview"])
    st.markdown(f"- {t['overview_desc']}")
    st.markdown(f"- {t['overview_desc2']}")

    summary = data["summary"]
    c1, c2, c3 = st.columns(3)
    c1.metric(t["current_run"], current_run.name)
    c2.metric(t["feature_rows"], f"{summary.get('rows', {}).get('feature_frame', 'N/A')}")
    c3.metric(t["pred_rows"], f"{summary.get('rows', {}).get('predictions', 'N/A')}")

    st.markdown(f"**{t['summary_json']}**")
    st.json(summary if summary else {})
    st.markdown(f"**{t['baseline_label']}:** `{baseline_run.name}`")



def show_pipeline(data: Dict[str, object], t: Dict[str, object]) -> None:
    st.subheader(t["pipeline"])
    st.info(t["pipeline_info"])
    for i, s in enumerate(t["pipeline_steps"], 1):
        st.markdown(f"{i}. {s}")

    col1, col2 = st.columns(2)
    col1.markdown(f"**{t['cluster_summary']}**")
    if data["cluster_summary"].empty:
        col1.info(t["missing_cluster"])
    else:
        col1.dataframe(data["cluster_summary"], use_container_width=True)

    col2.markdown(f"**{t['neighbor_top']}**")
    if data["neighbors"].empty:
        col2.info(t["missing_neighbor"])
    else:
        col2.dataframe(data["neighbors"].head(20), use_container_width=True)



def show_results(data: Dict[str, object], baseline_run: Path, t: Dict[str, object]) -> None:
    st.subheader(t["result"])
    st.info(t["result_info"])

    metrics = ensure_scenario(data["metrics"])
    if metrics.empty:
        st.warning(t["missing_generic"])
        return

    st.markdown(f"**{t['metrics_table']}**")
    c1, c2, c3 = st.columns(3)
    scenario = c1.multiselect("Scenario", sorted(metrics["scenario"].unique()), sorted(metrics["scenario"].unique()))
    segment = c2.multiselect("Segment", sorted(metrics["segment"].unique()), sorted(metrics["segment"].unique()))
    model = c3.multiselect("Model", sorted(metrics["model"].unique()), sorted(metrics["model"].unique()))
    view = metrics[
        metrics["scenario"].isin(scenario) & metrics["segment"].isin(segment) & metrics["model"].isin(model)
    ].sort_values(["scenario", "segment", "R2"], ascending=[True, True, False])
    st.dataframe(view, use_container_width=True)

    st.markdown(f"**{t['ablation_best']}**")
    ablation = data["ablation"]
    if ablation.empty:
        ablation = metrics.sort_values(["scenario", "segment", "R2"], ascending=[True, True, False]).groupby(["scenario", "segment"], as_index=False).head(1)
    st.dataframe(ablation, use_container_width=True)
    chart_df = ablation[["scenario", "segment", "R2"]].copy()
    chart_df["name"] = chart_df["scenario"] + " | " + chart_df["segment"]
    st.bar_chart(chart_df.set_index("name")["R2"], height=320)

    baseline_metrics = ensure_scenario(read_csv_safe(baseline_run / "metrics_all.csv"))
    if not baseline_metrics.empty:
        cur_full = metrics[metrics["scenario"] == "full"] if "full" in metrics["scenario"].unique() else metrics
        cur_best = cur_full.sort_values(["segment", "R2"], ascending=[True, False]).groupby("segment", as_index=False).head(1)
        base_best = baseline_metrics.sort_values(["segment", "R2"], ascending=[True, False]).groupby("segment", as_index=False).head(1)
        merged = cur_best[["segment", "model", "R2"]].merge(base_best[["segment", "model", "R2"]], on="segment", suffixes=("_current", "_baseline"))
        merged["delta_R2"] = merged["R2_current"] - merged["R2_baseline"]
        st.markdown(f"**{t['vs_baseline']}**")
        st.dataframe(merged, use_container_width=True)



def show_feature_insights(data: Dict[str, object], t: Dict[str, object]) -> None:
    st.subheader(t["feature"])
    st.info(t["feature_info"])

    st.markdown(f"**{t['spatial_fi']}**")
    spatial = data["spatial_fi"]
    if spatial.empty:
        st.warning(t["missing_generic"])
        return
    st.dataframe(spatial.head(100), use_container_width=True)

    focus = spatial.copy()
    if "scenario" in focus.columns and "full" in focus["scenario"].unique():
        focus = focus[focus["scenario"] == "full"]
    top = focus.sort_values("importance", ascending=False).head(20)
    top = top.assign(label=top["segment"] + " | " + top["feature"])
    st.bar_chart(top.set_index("label")["importance"], height=340)

    with st.expander(t["full_fi"]):
        fi = data["feature_importance"]
        if fi.empty:
            st.info(t["missing_generic"])
        else:
            st.dataframe(fi.head(200), use_container_width=True)



def show_predictions(data: Dict[str, object], t: Dict[str, object]) -> None:
    st.subheader(t["pred"])
    st.info(t["pred_info"])

    preds = ensure_scenario(data["predictions"])
    errs = data["error_analysis"]
    if preds.empty:
        st.warning(t["missing_generic"])
        return

    st.markdown(f"**{t['resid_img']}**")
    if data["residual_img"].exists():
        st.image(str(data["residual_img"]), use_container_width=True)
    else:
        st.info(t["missing_generic"])

    st.markdown(f"**{t['model_img']}**")
    if data["model_comparison_img"].exists():
        st.image(str(data["model_comparison_img"]), use_container_width=True)
    else:
        st.info(t["missing_generic"])

    st.markdown(f"**{t['view_pred']}**")
    c1, c2, c3 = st.columns(3)
    s = c1.selectbox("Scenario", sorted(preds["scenario"].unique()))
    g = c2.selectbox("Segment", sorted(preds["segment"].unique()))
    m = c3.selectbox("Model", sorted(preds["model"].unique()))
    st.dataframe(preds[(preds["scenario"] == s) & (preds["segment"] == g) & (preds["model"] == m)].head(300), use_container_width=True)

    st.markdown(f"**{t['err_station']}**")
    if errs.empty:
        st.info(t["missing_generic"])
    else:
        st.dataframe(errs.head(200), use_container_width=True)



def show_text(data: Dict[str, object], t: Dict[str, object]) -> None:
    st.subheader(t["explain"])
    st.info(t["explain_info"])

    st.markdown(f"**{t['conclusion']}**")
    if data["conclusion"]:
        st.markdown(data["conclusion"])
    else:
        st.info(t["missing_generic"])

    st.markdown(f"**{t['report']}**")
    if data["report"]:
        st.code(data["report"][:8000], language="markdown")
    else:
        st.info(t["missing_generic"])



def main() -> None:
    st.set_page_config(page_title="Casa0004 Dashboard", layout="wide")

    lang = st.sidebar.radio(I18N["zh"]["lang"], options=["中文", "English"], index=0)
    key = "zh" if lang == "中文" else "en"
    t = I18N[key]

    st.title(t["title"])

    runs = list_runs()
    if not runs:
        st.error(f"No runs found under: {OUTPUTS_DIR}")
        return

    run_names = [r.name for r in runs]
    default_idx = len(runs) - 1
    baseline_idx = run_names.index(DEFAULT_BASELINE) if DEFAULT_BASELINE in run_names else max(0, len(run_names) - 2)

    st.sidebar.header(t["run_select"])
    selected_name = st.sidebar.selectbox(t["current_run"], run_names, index=default_idx)
    baseline_name = st.sidebar.selectbox(t["baseline_run"], run_names, index=baseline_idx)

    selected_run = OUTPUTS_DIR / selected_name
    baseline_run = OUTPUTS_DIR / baseline_name

    if selected_run == baseline_run:
        st.sidebar.info(t["same_run"])

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{t['note']}**")
    st.sidebar.markdown(f"- {t['note1']}")
    st.sidebar.markdown(f"- {t['note2']}")

    data = load_run_artifacts(selected_run)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(t["tabs"])
    with tab1:
        show_overview(data, selected_run, baseline_run, t)
    with tab2:
        show_pipeline(data, t)
    with tab3:
        show_results(data, baseline_run, t)
    with tab4:
        show_feature_insights(data, t)
    with tab5:
        show_predictions(data, t)
    with tab6:
        show_text(data, t)


if __name__ == "__main__":
    main()
