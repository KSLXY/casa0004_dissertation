from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"
DEFAULT_BASELINE = "run_20260415_102043"


@st.cache_data(show_spinner=False)
def list_runs() -> List[Path]:
    runs = sorted([p for p in OUTPUTS_DIR.glob("run_*") if p.is_dir()])
    return runs


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
        "cluster_map": read_csv_safe(run_dir / "cluster_artifacts" / "station_cluster_map.csv"),
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
        df = df.copy()
        df["scenario"] = "legacy"
    return df



def show_overview(current_run: Path, data: Dict[str, object], baseline_run: Path | None) -> None:
    st.subheader("项目概览")
    st.markdown(
        """
- 本看板展示 Casa0004 的可复现建模流程：数据接入 → 特征工程 → 空间增强（聚类 + 邻居）→ 事件增强 → 消融实验 → 结果解释。
- 当前页面重点用于对比 **本次增强 run** 与 **历史基线 run** 的效果差异（主指标 R²）。
        """
    )

    summary = data["summary"]
    c1, c2, c3 = st.columns(3)
    c1.metric("当前 Run", current_run.name)
    c2.metric("特征行数", f"{summary.get('rows', {}).get('feature_frame', 'N/A')}")
    c3.metric("预测行数", f"{summary.get('rows', {}).get('predictions', 'N/A')}")

    if summary:
        st.markdown("**当前 run 摘要（run_summary.json）**")
        st.json(summary)

    if baseline_run is not None:
        st.markdown(f"**基线 Run:** `{baseline_run.name}`")



def show_pipeline_process(data: Dict[str, object]) -> None:
    st.subheader("流程与工件")
    st.markdown(
        """
1. `data_ingest`: 校验输入 schema，生成 `data_version.json`
2. `cluster_stage`: 自动搜索最佳 k，输出 `cluster_artifacts/*`
3. `neighbor_stage`: 构建 `station_neighbors.csv`，生成邻居滞后特征
4. `event_stage`: 合并节假日/学校假期/大型活动特征
5. `train`: weekday/weekend × 5个scenario × 3个模型
6. `evaluate/report`: 产出 `metrics_all.csv`、`ablation_metrics.csv`、图表和结论草稿
        """
    )

    left, right = st.columns(2)
    cluster_summary = data["cluster_summary"]
    neighbors = data["neighbors"]

    left.markdown("**聚类摘要**")
    if not cluster_summary.empty:
        left.dataframe(cluster_summary, use_container_width=True)
    else:
        left.info("当前 run 未找到 cluster_summary.csv")

    right.markdown("**邻接关系（前20行）**")
    if not neighbors.empty:
        right.dataframe(neighbors.head(20), use_container_width=True)
    else:
        right.info("当前 run 未找到 station_neighbors.csv")



def show_metrics_and_ablation(current_run: Path, data: Dict[str, object], baseline_run: Path | None) -> None:
    st.subheader("结果与消融")

    metrics = ensure_scenario(data["metrics"])
    if metrics.empty:
        st.warning("当前 run 没有 metrics_all.csv")
        return

    st.markdown("**完整指标表（可筛选）**")
    scenario_options = sorted(metrics["scenario"].unique().tolist())
    seg_options = sorted(metrics["segment"].unique().tolist())
    model_options = sorted(metrics["model"].unique().tolist())

    c1, c2, c3 = st.columns(3)
    pick_scenario = c1.multiselect("Scenario", scenario_options, default=scenario_options)
    pick_seg = c2.multiselect("Segment", seg_options, default=seg_options)
    pick_model = c3.multiselect("Model", model_options, default=model_options)

    view = metrics[
        metrics["scenario"].isin(pick_scenario)
        & metrics["segment"].isin(pick_seg)
        & metrics["model"].isin(pick_model)
    ].copy()

    st.dataframe(view.sort_values(["scenario", "segment", "R2"], ascending=[True, True, False]), use_container_width=True)

    st.markdown("**Ablation 最优结果（每个 scenario × segment）**")
    ablation = data["ablation"]
    if ablation.empty:
        ablation = (
            metrics.sort_values(["scenario", "segment", "R2"], ascending=[True, True, False])
            .groupby(["scenario", "segment"], as_index=False)
            .head(1)
        )
    st.dataframe(ablation, use_container_width=True)

    # Visual R2 comparison
    chart_df = ablation[["scenario", "segment", "R2"]].copy()
    chart_df["name"] = chart_df["scenario"] + " | " + chart_df["segment"]
    st.bar_chart(chart_df.set_index("name")["R2"], height=340)

    # Baseline comparison by best full scenario
    if baseline_run is not None:
        baseline_metrics = ensure_scenario(read_csv_safe(baseline_run / "metrics_all.csv"))
        if not baseline_metrics.empty:
            cur_full = metrics[metrics["scenario"] == "full"] if "full" in metrics["scenario"].unique() else metrics
            cur_best = cur_full.sort_values(["segment", "R2"], ascending=[True, False]).groupby("segment", as_index=False).head(1)
            base_best = baseline_metrics.sort_values(["segment", "R2"], ascending=[True, False]).groupby("segment", as_index=False).head(1)
            merged = cur_best[["segment", "model", "R2"]].merge(
                base_best[["segment", "model", "R2"]],
                on="segment",
                suffixes=("_current", "_baseline"),
            )
            merged["delta_R2"] = merged["R2_current"] - merged["R2_baseline"]
            st.markdown("**与基线对照（最佳模型）**")
            st.dataframe(merged, use_container_width=True)



def show_feature_insights(data: Dict[str, object]) -> None:
    st.subheader("特征解释")
    spatial = data["spatial_fi"]
    all_fi = data["feature_importance"]

    st.markdown("**空间/事件特征重要性（spatial_feature_importance.csv）**")
    if not spatial.empty:
        st.dataframe(spatial.head(80), use_container_width=True)

        focus = spatial.copy()
        if "scenario" in focus.columns and "full" in focus["scenario"].unique():
            focus = focus[focus["scenario"] == "full"]
        top = focus.sort_values("importance", ascending=False).head(20)
        top = top.assign(label=top["segment"] + " | " + top["feature"])
        st.bar_chart(top.set_index("label")["importance"], height=360)
    else:
        st.info("当前 run 未找到 spatial_feature_importance.csv")

    with st.expander("查看完整 feature_importance.csv"):
        if not all_fi.empty:
            st.dataframe(all_fi.head(200), use_container_width=True)
        else:
            st.info("当前 run 未找到 feature_importance.csv")



def show_predictions_and_errors(data: Dict[str, object]) -> None:
    st.subheader("预测与误差")
    preds = data["predictions"]
    errs = data["error_analysis"]

    if preds.empty:
        st.warning("当前 run 没有 predictions.csv")
        return

    preds = ensure_scenario(preds)

    st.markdown("**残差分布图片（pipeline 自动生成）**")
    img = data["residual_img"]
    if img.exists():
        st.image(str(img), use_container_width=True)
    else:
        st.info("未找到 residual_distribution.png")

    st.markdown("**模型对比图片（pipeline 自动生成）**")
    img2 = data["model_comparison_img"]
    if img2.exists():
        st.image(str(img2), use_container_width=True)
    else:
        st.info("未找到 model_comparison.png")

    st.markdown("**按场景/分段/模型查看预测样本**")
    c1, c2, c3 = st.columns(3)
    s = c1.selectbox("Scenario", sorted(preds["scenario"].unique()))
    g = c2.selectbox("Segment", sorted(preds["segment"].unique()))
    m = c3.selectbox("Model", sorted(preds["model"].unique()))

    view = preds[(preds["scenario"] == s) & (preds["segment"] == g) & (preds["model"] == m)]
    st.dataframe(view.head(300), use_container_width=True)

    st.markdown("**误差汇总（站点粒度）**")
    if not errs.empty:
        st.dataframe(errs.head(200), use_container_width=True)



def show_explanation_text(data: Dict[str, object]) -> None:
    st.subheader("文字解释与结论")

    st.markdown("**结论草稿（conclusion_draft.md）**")
    conclusion = data["conclusion"]
    if conclusion:
        st.markdown(conclusion)
    else:
        st.info("当前 run 未找到 conclusion_draft.md")

    st.markdown("**报告摘要（report.md）**")
    report = data["report"]
    if report:
        st.code(report[:8000], language="markdown")
    else:
        st.info("当前 run 未找到 report.md")



def main() -> None:
    st.set_page_config(page_title="Casa0004 可视化看板", layout="wide")
    st.title("Casa0004 Pipeline 可视化看板")

    runs = list_runs()
    if not runs:
        st.error(f"未找到任何 run 目录：{OUTPUTS_DIR}")
        return

    run_names = [r.name for r in runs]
    default_idx = len(runs) - 1

    st.sidebar.header("运行选择")
    selected_name = st.sidebar.selectbox("当前 Run", run_names, index=default_idx)
    selected_run = OUTPUTS_DIR / selected_name

    baseline_idx = run_names.index(DEFAULT_BASELINE) if DEFAULT_BASELINE in run_names else max(0, len(run_names) - 2)
    baseline_name = st.sidebar.selectbox("基线 Run", run_names, index=baseline_idx)
    baseline_run = OUTPUTS_DIR / baseline_name

    if selected_run == baseline_run:
        st.sidebar.info("当前 run 与基线 run 相同，可切换以查看提升对比。")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**说明**")
    st.sidebar.markdown("- 旧版 run 缺少 `ablation/spatial` 文件时会自动降级显示")
    st.sidebar.markdown("- 指标主比较维度：R²")

    data = load_run_artifacts(selected_run)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["概览", "流程工件", "结果消融", "特征解释", "预测误差", "文字结论"]
    )

    with tab1:
        show_overview(selected_run, data, baseline_run)
    with tab2:
        show_pipeline_process(data)
    with tab3:
        show_metrics_and_ablation(selected_run, data, baseline_run)
    with tab4:
        show_feature_insights(data)
    with tab5:
        show_predictions_and_errors(data)
    with tab6:
        show_explanation_text(data)


if __name__ == "__main__":
    main()
