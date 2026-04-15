from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"


TEXT: Dict[str, Dict[str, Any]] = {
    "zh": {
        "page_title": "Casa0004 可视化看板",
        "app_title": "Casa0004 共享单车需求研究",
        "language": "语言",
        "language_options": ["中文", "English"],
        "guide_title": "阅读导览",
        "guide_items": [
            "页面按背景、数据流程、模型对比、原因解释、误差边界、总结建议组织",
            "同一口径用于全部页面，避免指标定义在不同页面发生变化",
            "若旧版结果缺少部分文件，页面会自动使用可用数据降级展示",
        ],
        "tab_overview": "背景与目标",
        "tab_process": "数据与流程",
        "tab_compare": "模型方案对比",
        "tab_features": "影响因素解释",
        "tab_error": "预测误差分析",
        "tab_conclusion": "总结与改进",
        "summary_title": "核心说明",
        "overview_summary": "研究对象是都柏林共享单车的站点小时需求。目标是建立可复跑流程，用于支持调度和资源配置。",
        "overview_points": [
            "需求会随时段、天气、站点位置变化，单一因素难以充分解释",
            "流程强调可复现，便于重复验证并支持论文与展示材料同步更新",
            "当前页面展示最新一次运行结果",
        ],
        "metric_run": "结果版本",
        "metric_rows_feature": "特征记录数",
        "metric_rows_pred": "预测记录数",
        "run_summary": "运行摘要",
        "process_summary": "模型输入来自天气、站点流量、空间邻接和事件信息。各阶段输出固定文件，便于复查与复跑。",
        "process_points": [
            "数据接入阶段校验字段并记录版本信息",
            "空间阶段生成站点聚类与邻近站点关系",
            "事件阶段合并法定假日、学校假期和大型活动信息",
            "训练阶段按工作日与周末分别建模并输出统一指标",
        ],
        "cluster_table": "聚类摘要",
        "neighbor_table": "邻接关系示例 前20行",
        "missing_cluster": "未找到 cluster_summary 文件",
        "missing_neighbor": "未找到 station_neighbors 文件",
        "large_file_title": "大文件接入建议",
        "large_file_items": [
            "优先使用 parquet 或 csv.gz，体积更小，读取更快",
            "结果文件建议按 run 目录分版本保存，避免单文件无限增长",
            "在线部署场景建议将超大原始数据放在对象存储，仅拉取建模所需字段",
        ],
        "compare_summary": "该页比较不同特征方案在同一时间切分下的表现。比较重点是 R²，也同时保留 MAE 与 RMSE。",
        "compare_points": [
            "对比使用统一训练和测试切分，确保可比性",
            "方案增量对比用于识别提升来自哪些新增信息",
            "若无增量对比文件，可由完整指标表自动汇总生成",
        ],
        "missing_metrics": "缺少 metrics_all 文件，无法展示模型对比",
        "filter_scenario": "方案",
        "filter_segment": "分段",
        "filter_model": "模型",
        "metrics_table": "完整指标表",
        "incremental_table": "方案增量对比表",
        "incremental_source_note": "方案增量对比来自可复现重构流程。方法是在同一数据切分下，按特征组合逐组训练并比较指标。",
        "incremental_need_title": "是否有必要保留该模块",
        "incremental_need_text": "若仅展示单个最终模型，该模块可以省略。若需要解释改进来源并支持论文论证，建议保留。该模块可回答两个问题，即提升是否真实存在，以及提升主要来自哪一类特征。",
        "features_summary": "该页回答模型为何产生当前表现。通过重要性排序观察时间、空间和事件变量的相对贡献。",
        "features_points": [
            "重要性用于描述模型依赖程度，不直接代表因果关系",
            "不同分段下重要性会变化，反映出场景差异",
            "可结合业务经验判断特征是否可解释且可行动",
        ],
        "spatial_fi": "空间与事件特征重要性",
        "missing_spatial_fi": "未找到 spatial_feature_importance 文件",
        "terms_title": "术语补充与完整特征表",
        "terms_text": "R² 表示拟合程度，数值越高通常越好。方案增量对比指在同一口径下逐组加入特征并比较效果。特征重要性用于观察模型依赖关系，不等同因果结论。",
        "missing_fi": "未找到 feature_importance 文件",
        "error_summary": "该页展示预测误差在样本与站点层面的分布情况，用于识别稳定区间与高风险区间。",
        "error_points": [
            "优先查看残差分布和站点级误差汇总",
            "若缺少 error_analysis 文件，系统会根据 predictions 自动回算",
            "误差分析可直接连接到后续优化策略，例如补充事件信息或细化时段特征",
        ],
        "missing_predictions": "缺少 predictions 文件，无法展示误差分析",
        "residual_img": "残差分布图",
        "model_img": "模型对比图",
        "missing_residual_img": "未找到 residual_distribution 图",
        "missing_model_img": "未找到 model_comparison 图",
        "pred_sample": "预测样本预览",
        "error_table": "误差汇总 站点粒度",
        "error_derived": "未检测到 error_analysis 文件，当前表格由 predictions 自动回算生成。",
        "conclusion_summary": "该页汇总主要发现、可复现状态与后续改进方向。",
        "conclusion_points": [
            "当前流程已具备一键复跑能力",
            "空间与事件信息已接入统一训练口径",
            "后续可继续细化事件库与时空建模策略",
        ],
        "conclusion_draft": "结论草稿",
        "missing_conclusion": "未找到 conclusion_draft 文件",
        "report_summary": "报告摘要",
        "missing_report": "未找到 report 文件",
        "no_run": "未找到 run 目录",
    },
    "en": {
        "page_title": "Casa0004 Dashboard",
        "app_title": "Casa0004 Bike Demand Study",
        "language": "Language",
        "language_options": ["中文", "English"],
        "guide_title": "Reading Guide",
        "guide_items": [
            "Pages are organized as background, data flow, model comparison, interpretation, error boundary, and conclusion",
            "A shared metric definition is applied across pages to keep results consistent",
            "If older runs miss some files, the page falls back to available artifacts",
        ],
        "tab_overview": "Background",
        "tab_process": "Data Flow",
        "tab_compare": "Model Comparison",
        "tab_features": "Driver Insights",
        "tab_error": "Error Analysis",
        "tab_conclusion": "Conclusion",
        "summary_title": "Core Summary",
        "overview_summary": "The task is hourly station demand prediction for Dublin bike sharing. The goal is a reproducible workflow that supports operations and reporting.",
        "overview_points": [
            "Demand changes with time, weather, and station location",
            "The workflow is built for reruns and transparent validation",
            "This page shows the latest available run",
        ],
        "metric_run": "Run Version",
        "metric_rows_feature": "Feature Rows",
        "metric_rows_pred": "Prediction Rows",
        "run_summary": "Run Summary",
        "process_summary": "Model inputs combine weather, station flow, spatial relations, and event signals. Each stage writes fixed artifacts for traceability.",
        "process_points": [
            "Ingestion validates schema and records data version",
            "Spatial stage builds clusters and station neighbors",
            "Event stage merges holidays, school breaks, and major events",
            "Training runs separate models for weekday and weekend",
        ],
        "cluster_table": "Cluster Summary",
        "neighbor_table": "Neighbor Sample Top 20",
        "missing_cluster": "cluster_summary file not found",
        "missing_neighbor": "station_neighbors file not found",
        "large_file_title": "Large File Guidance",
        "large_file_items": [
            "Prefer parquet or csv.gz for smaller size and faster reads",
            "Store outputs by run directory instead of one growing file",
            "For cloud deployment, keep raw large files in object storage and load only required columns",
        ],
        "compare_summary": "This page compares feature-set scenarios under the same split protocol. R² is the main metric, with MAE and RMSE retained.",
        "compare_points": [
            "All scenarios share the same train and test split",
            "Incremental scenario comparison shows where gains come from",
            "If no incremental file exists, the table is derived from metrics_all",
        ],
        "missing_metrics": "metrics_all file is missing",
        "filter_scenario": "Scenario",
        "filter_segment": "Segment",
        "filter_model": "Model",
        "metrics_table": "Metrics Table",
        "incremental_table": "Incremental Scenario Table",
        "incremental_source_note": "Incremental comparison is produced by the reproducible pipeline. It trains multiple feature-set scenarios under the same split and compares metrics.",
        "incremental_need_title": "Should this module stay",
        "incremental_need_text": "This module can be removed when only one final model is needed. It is recommended when the report must explain where the improvement comes from and whether the gain is robust.",
        "features_summary": "This page explains why the model behaves as observed. Importance rankings summarize relative contribution by variable groups.",
        "features_points": [
            "Importance reflects model reliance, not direct causality",
            "Importance may differ between weekday and weekend",
            "Business context is needed for interpretation",
        ],
        "spatial_fi": "Spatial and Event Feature Importance",
        "missing_spatial_fi": "spatial_feature_importance file not found",
        "terms_title": "Terms and Full Feature Table",
        "terms_text": "R² indicates fit quality and is usually better when higher. Incremental comparison means adding feature groups one by one under identical settings. Importance describes dependence, not causation.",
        "missing_fi": "feature_importance file not found",
        "error_summary": "This page shows error distribution at sample and station levels to identify stable and unstable areas.",
        "error_points": [
            "Use residual distribution and station error summary first",
            "If error_analysis is missing, it is computed from predictions",
            "Error patterns guide next feature improvements",
        ],
        "missing_predictions": "predictions file is missing",
        "residual_img": "Residual Distribution",
        "model_img": "Model Comparison",
        "missing_residual_img": "residual_distribution image not found",
        "missing_model_img": "model_comparison image not found",
        "pred_sample": "Prediction Sample",
        "error_table": "Station-level Error Summary",
        "error_derived": "error_analysis file is missing. The table below is computed from predictions.",
        "conclusion_summary": "This page consolidates findings, reproducibility status, and next actions.",
        "conclusion_points": [
            "The pipeline supports one-command reruns",
            "Spatial and event features are integrated in one protocol",
            "Further gains may come from richer event coverage and stronger spatiotemporal models",
        ],
        "conclusion_draft": "Conclusion Draft",
        "missing_conclusion": "conclusion_draft file not found",
        "report_summary": "Report Excerpt",
        "missing_report": "report file not found",
        "no_run": "No run directory found",
    },
}


def tr(lang: str, key: str) -> Any:
    return TEXT[lang][key]


@st.cache_data(show_spinner=False)
def list_runs() -> List[Path]:
    return sorted([p for p in OUTPUTS_DIR.glob("run_*") if p.is_dir()])


@st.cache_data(show_spinner=False)
def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_parquet_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def read_json_safe(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_text_safe(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def read_table_flexible(base: Path) -> pd.DataFrame:
    csv_path = base.with_suffix(".csv")
    gz_path = base.with_suffix(".csv.gz")
    pq_path = base.with_suffix(".parquet")

    if csv_path.exists():
        return read_csv_safe(csv_path)
    if gz_path.exists():
        return read_csv_safe(gz_path)
    if pq_path.exists():
        return read_parquet_safe(pq_path)
    return pd.DataFrame()


def load_run_artifacts(run_dir: Path) -> Dict[str, Any]:
    return {
        "metrics": read_table_flexible(run_dir / "metrics_all"),
        "ablation": read_table_flexible(run_dir / "ablation_metrics"),
        "feature_importance": read_table_flexible(run_dir / "feature_importance"),
        "spatial_fi": read_table_flexible(run_dir / "spatial_feature_importance"),
        "error_analysis": read_table_flexible(run_dir / "error_analysis"),
        "predictions": read_table_flexible(run_dir / "predictions"),
        "neighbors": read_table_flexible(run_dir / "station_neighbors"),
        "cluster_summary": read_table_flexible(run_dir / "cluster_artifacts" / "cluster_summary"),
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


def render_summary_block(lang: str, summary: str, points: List[str]) -> None:
    st.markdown(f"**{tr(lang, 'summary_title')}**  {summary}")
    for item in points:
        st.markdown(f"- {item}")


def build_error_analysis_from_predictions(preds: pd.DataFrame) -> pd.DataFrame:
    required = {"scenario", "segment", "model", "station_id", "y_true", "y_pred"}
    if preds.empty or not required.issubset(set(preds.columns)):
        return pd.DataFrame()

    df = preds.copy()
    if "abs_residual" not in df.columns:
        df["abs_residual"] = (df["y_true"] - df["y_pred"]).abs()
    sq = (df["y_true"] - df["y_pred"]) ** 2
    df["sq_error"] = sq

    grouped = (
        df.groupby(["scenario", "segment", "model", "station_id"], as_index=False)
        .agg(
            n=("y_true", "size"),
            mae=("abs_residual", "mean"),
            mse=("sq_error", "mean"),
            mean_true=("y_true", "mean"),
            mean_pred=("y_pred", "mean"),
        )
    )
    grouped["rmse"] = grouped["mse"] ** 0.5
    return grouped.drop(columns=["mse"])


def show_overview(run_dir: Path, data: Dict[str, Any], lang: str) -> None:
    st.subheader(tr(lang, "tab_overview"))
    render_summary_block(lang, tr(lang, "overview_summary"), tr(lang, "overview_points"))

    summary = data["summary"]
    c1, c2, c3 = st.columns(3)
    c1.metric(tr(lang, "metric_run"), run_dir.name)
    c2.metric(tr(lang, "metric_rows_feature"), f"{summary.get('rows', {}).get('feature_frame', 'N/A')}")
    c3.metric(tr(lang, "metric_rows_pred"), f"{summary.get('rows', {}).get('predictions', 'N/A')}")

    if summary:
        st.markdown(f"**{tr(lang, 'run_summary')}**")
        st.json(summary)


def show_pipeline_process(data: Dict[str, Any], lang: str) -> None:
    st.subheader(tr(lang, "tab_process"))
    render_summary_block(lang, tr(lang, "process_summary"), tr(lang, "process_points"))

    left, right = st.columns(2)
    cluster_summary = data["cluster_summary"]
    neighbors = data["neighbors"]

    left.markdown(f"**{tr(lang, 'cluster_table')}**")
    if not cluster_summary.empty:
        left.dataframe(cluster_summary, use_container_width=True)
    else:
        left.info(tr(lang, "missing_cluster"))

    right.markdown(f"**{tr(lang, 'neighbor_table')}**")
    if not neighbors.empty:
        right.dataframe(neighbors.head(20), use_container_width=True)
    else:
        right.info(tr(lang, "missing_neighbor"))

    st.markdown(f"**{tr(lang, 'large_file_title')}**")
    for item in tr(lang, "large_file_items"):
        st.markdown(f"- {item}")


def show_metrics_comparison(data: Dict[str, Any], lang: str) -> None:
    st.subheader(tr(lang, "tab_compare"))
    render_summary_block(lang, tr(lang, "compare_summary"), tr(lang, "compare_points"))

    metrics = ensure_scenario(data["metrics"])
    if metrics.empty:
        st.warning(tr(lang, "missing_metrics"))
        return

    st.markdown(f"**{tr(lang, 'metrics_table')}**")
    scenario_options = sorted(metrics["scenario"].dropna().unique().tolist())
    segment_options = sorted(metrics["segment"].dropna().unique().tolist())
    model_options = sorted(metrics["model"].dropna().unique().tolist())

    c1, c2, c3 = st.columns(3)
    pick_scenario = c1.multiselect(tr(lang, "filter_scenario"), scenario_options, default=scenario_options)
    pick_segment = c2.multiselect(tr(lang, "filter_segment"), segment_options, default=segment_options)
    pick_model = c3.multiselect(tr(lang, "filter_model"), model_options, default=model_options)

    view = metrics[
        metrics["scenario"].isin(pick_scenario)
        & metrics["segment"].isin(pick_segment)
        & metrics["model"].isin(pick_model)
    ].copy()

    if not view.empty:
        st.dataframe(
            view.sort_values(["scenario", "segment", "R2"], ascending=[True, True, False]),
            use_container_width=True,
        )

    st.markdown(f"**{tr(lang, 'incremental_table')}**")
    st.caption(tr(lang, "incremental_source_note"))
    incremental = data["ablation"]
    if incremental.empty:
        incremental = (
            metrics.sort_values(["scenario", "segment", "R2"], ascending=[True, True, False])
            .groupby(["scenario", "segment"], as_index=False)
            .head(1)
        )
    st.dataframe(incremental, use_container_width=True)

    if {"scenario", "segment", "R2"}.issubset(incremental.columns):
        chart_df = incremental[["scenario", "segment", "R2"]].copy()
        chart_df["name"] = chart_df["scenario"] + " | " + chart_df["segment"]
        st.bar_chart(chart_df.set_index("name")["R2"], height=320)

    with st.expander(tr(lang, "incremental_need_title")):
        st.markdown(tr(lang, "incremental_need_text"))


def show_feature_insights(data: Dict[str, Any], lang: str) -> None:
    st.subheader(tr(lang, "tab_features"))
    render_summary_block(lang, tr(lang, "features_summary"), tr(lang, "features_points"))

    spatial = data["spatial_fi"]
    all_fi = data["feature_importance"]

    st.markdown(f"**{tr(lang, 'spatial_fi')}**")
    if not spatial.empty:
        st.dataframe(spatial.head(80), use_container_width=True)

        focus = spatial.copy()
        if "scenario" in focus.columns and "full" in focus["scenario"].unique():
            focus = focus[focus["scenario"] == "full"]
        if {"importance", "segment", "feature"}.issubset(focus.columns):
            top = focus.sort_values("importance", ascending=False).head(20)
            top = top.assign(label=top["segment"].astype(str) + " | " + top["feature"].astype(str))
            st.bar_chart(top.set_index("label")["importance"], height=340)
    else:
        st.info(tr(lang, "missing_spatial_fi"))

    with st.expander(tr(lang, "terms_title")):
        st.markdown(tr(lang, "terms_text"))
        if not all_fi.empty:
            st.dataframe(all_fi.head(200), use_container_width=True)
        else:
            st.info(tr(lang, "missing_fi"))


def show_predictions_and_errors(data: Dict[str, Any], lang: str) -> None:
    st.subheader(tr(lang, "tab_error"))
    render_summary_block(lang, tr(lang, "error_summary"), tr(lang, "error_points"))

    preds = ensure_scenario(data["predictions"])
    if preds.empty:
        st.warning(tr(lang, "missing_predictions"))
        return

    st.markdown(f"**{tr(lang, 'residual_img')}**")
    residual_img = data["residual_img"]
    if residual_img.exists():
        st.image(str(residual_img), use_container_width=True)
    else:
        st.info(tr(lang, "missing_residual_img"))

    st.markdown(f"**{tr(lang, 'model_img')}**")
    model_img = data["model_comparison_img"]
    if model_img.exists():
        st.image(str(model_img), use_container_width=True)
    else:
        st.info(tr(lang, "missing_model_img"))

    st.markdown(f"**{tr(lang, 'pred_sample')}**")
    c1, c2, c3 = st.columns(3)
    scenarios = sorted(preds["scenario"].dropna().unique().tolist())
    segments = sorted(preds["segment"].dropna().unique().tolist())
    models = sorted(preds["model"].dropna().unique().tolist())

    s = c1.selectbox(tr(lang, "filter_scenario"), scenarios)
    g = c2.selectbox(tr(lang, "filter_segment"), segments)
    m = c3.selectbox(tr(lang, "filter_model"), models)

    view = preds[(preds["scenario"] == s) & (preds["segment"] == g) & (preds["model"] == m)]
    st.dataframe(view.head(300), use_container_width=True)

    errs = data["error_analysis"]
    if errs.empty:
        errs = build_error_analysis_from_predictions(preds)
        if not errs.empty:
            st.info(tr(lang, "error_derived"))

    st.markdown(f"**{tr(lang, 'error_table')}**")
    if not errs.empty:
        st.dataframe(errs.head(200), use_container_width=True)


def show_conclusion(data: Dict[str, Any], lang: str) -> None:
    st.subheader(tr(lang, "tab_conclusion"))
    render_summary_block(lang, tr(lang, "conclusion_summary"), tr(lang, "conclusion_points"))

    st.markdown(f"**{tr(lang, 'conclusion_draft')}**")
    conclusion = data["conclusion"]
    if conclusion:
        st.markdown(conclusion)
    else:
        st.info(tr(lang, "missing_conclusion"))

    st.markdown(f"**{tr(lang, 'report_summary')}**")
    report = data["report"]
    if report:
        st.code(report[:8000], language="markdown")
    else:
        st.info(tr(lang, "missing_report"))


def main() -> None:
    st.set_page_config(page_title="Casa0004 Dashboard", layout="wide")

    runs = list_runs()
    if not runs:
        st.error(f"{tr('zh', 'no_run')} {OUTPUTS_DIR}")
        return

    selected_language = st.radio(
        "Language 语言",
        options=TEXT["zh"]["language_options"],
        horizontal=True,
        index=0,
    )
    lang = "zh" if selected_language == "中文" else "en"

    st.title(tr(lang, "app_title"))
    st.markdown(f"**{tr(lang, 'guide_title')}**")
    for item in tr(lang, "guide_items"):
        st.markdown(f"- {item}")

    selected_run = runs[-1]
    data = load_run_artifacts(selected_run)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            tr(lang, "tab_overview"),
            tr(lang, "tab_process"),
            tr(lang, "tab_compare"),
            tr(lang, "tab_features"),
            tr(lang, "tab_error"),
            tr(lang, "tab_conclusion"),
        ]
    )

    with tab1:
        show_overview(selected_run, data, lang)
    with tab2:
        show_pipeline_process(data, lang)
    with tab3:
        show_metrics_comparison(data, lang)
    with tab4:
        show_feature_insights(data, lang)
    with tab5:
        show_predictions_and_errors(data, lang)
    with tab6:
        show_conclusion(data, lang)


if __name__ == "__main__":
    main()
