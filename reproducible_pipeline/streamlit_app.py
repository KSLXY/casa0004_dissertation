from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"


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



def render_story_header(prev_text: str, core_text: str, next_text: str) -> None:
    st.markdown(prev_text)
    st.markdown(f"**本页核心结论**  {core_text}")
    st.caption(next_text)


def show_overview(current_run: Path, data: Dict[str, object]) -> None:
    st.subheader("项目概览")
    render_story_header(
        "这一步用于说明研究背景，并建立后续分析的阅读框架。",
        "共享单车需求会随时段、天气与空间位置波动，稳定预测可支持调度与资源配置。",
        "下一步进入数据处理流程，说明原始记录如何转化为可训练特征。",
    )
    st.markdown(
        """
本看板展示 Casa0004 的可复现建模流程。流程覆盖数据接入、特征构建、空间增强、事件增强、效果评估与结果解释。  
目标是将分散的分析步骤整理为可复跑、可复核、可解释的一条链路。
        """
    )

    summary = data["summary"]
    c1, c2, c3 = st.columns(3)
    c1.metric("当前版本", current_run.name)
    c2.metric("特征行数", f"{summary.get('rows', {}).get('feature_frame', 'N/A')}")
    c3.metric("预测行数", f"{summary.get('rows', {}).get('predictions', 'N/A')}")

    if summary:
        st.markdown("**运行摘要 `run_summary.json`**")
        st.json(summary)



def show_pipeline_process(data: Dict[str, object]) -> None:
    st.subheader("流程与工件")
    render_story_header(
        "上一页给出研究目标，这一步说明实现路径与关键中间产物。",
        "同一份原始数据会按固定顺序完成清洗、空间构建、事件合并与训练评估，结果可以重复得到。",
        "下一步查看量化结果，判断不同方案带来的性能变化。",
    )
    st.markdown(
        """
1. `data_ingest` 统一输入结构，并生成 `data_version.json` 用于版本记录  
2. `cluster_stage` 搜索合适簇数，并输出 `cluster_artifacts/*`  
3. `neighbor_stage` 构建 `station_neighbors.csv`，提取邻近站点滞后流量  
4. `event_stage` 合并法定假日、学校假期与重点活动信息  
5. `train` 按工作日与周末分别训练多组模型  
6. `evaluate` 与 `report` 输出指标文件、图表与结论草稿
        """
    )

    left, right = st.columns(2)
    cluster_summary = data["cluster_summary"]
    neighbors = data["neighbors"]

    left.markdown("**聚类结果摘要**")
    if not cluster_summary.empty:
        left.dataframe(cluster_summary, use_container_width=True)
    else:
        left.info("当前版本未找到 `cluster_summary.csv`")

    right.markdown("**邻接关系示例 前20行**")
    if not neighbors.empty:
        right.dataframe(neighbors.head(20), use_container_width=True)
    else:
        right.info("当前版本未找到 `station_neighbors.csv`")



def show_metrics_and_ablation(data: Dict[str, object]) -> None:
    st.subheader("结果与消融")
    render_story_header(
        "上一页说明了数据如何进入模型，这一步集中展示训练结果。",
        "分场景消融可识别哪些新增信息确实提升了预测质量。",
        "下一步解释提升来自哪些特征，并说明其现实含义。",
    )

    metrics = ensure_scenario(data["metrics"])
    if metrics.empty:
        st.warning("当前版本缺少 `metrics_all.csv`")
        return

    st.markdown("**完整指标表 可筛选查看**")
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

    st.markdown("**消融最优结果 按 scenario 与 segment 汇总**")
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



def show_feature_insights(data: Dict[str, object]) -> None:
    st.subheader("特征解释")
    render_story_header(
        "上一页展示了结果差异，这一步解释差异形成的原因。",
        "空间与事件特征在不同分段中的重要性不同，提示需求变化具有明显场景依赖。",
        "下一步查看误差分布，识别模型稳定与不稳定的位置。",
    )
    spatial = data["spatial_fi"]
    all_fi = data["feature_importance"]

    st.markdown("**空间与事件特征重要性 `spatial_feature_importance.csv`**")
    if not spatial.empty:
        st.dataframe(spatial.head(80), use_container_width=True)

        focus = spatial.copy()
        if "scenario" in focus.columns and "full" in focus["scenario"].unique():
            focus = focus[focus["scenario"] == "full"]
        top = focus.sort_values("importance", ascending=False).head(20)
        top = top.assign(label=top["segment"] + " | " + top["feature"])
        st.bar_chart(top.set_index("label")["importance"], height=360)
    else:
        st.info("当前版本未找到 `spatial_feature_importance.csv`")

    with st.expander("术语补充与完整特征表"):
        st.markdown(
            """
`R²` 用于衡量预测与真实值的一致程度，数值越高代表拟合越好。  
`消融` 是逐项加入新特征并比较效果的过程，可用于判断改进来源。  
`特征重要性` 反映模型在当前数据中对变量的依赖程度，不直接代表因果关系。
            """
        )
        if not all_fi.empty:
            st.dataframe(all_fi.head(200), use_container_width=True)
        else:
            st.info("当前版本未找到 `feature_importance.csv`")



def show_predictions_and_errors(data: Dict[str, object]) -> None:
    st.subheader("预测与误差")
    render_story_header(
        "上一页解释了模型为什么有效，这一步说明模型在哪些位置更稳。",
        "误差分布可用于识别高波动时段和难预测站点，为后续优化提供方向。",
        "下一步给出完整结论，并整理可以直接落地的改进建议。",
    )
    preds = data["predictions"]
    errs = data["error_analysis"]

    if preds.empty:
        st.warning("当前版本缺少 `predictions.csv`")
        return

    preds = ensure_scenario(preds)

    st.markdown("**残差分布图 自动生成**")
    img = data["residual_img"]
    if img.exists():
        st.image(str(img), use_container_width=True)
    else:
        st.info("未找到 `residual_distribution.png`")

    st.markdown("**模型对比图 自动生成**")
    img2 = data["model_comparison_img"]
    if img2.exists():
        st.image(str(img2), use_container_width=True)
    else:
        st.info("未找到 `model_comparison.png`")

    st.markdown("**按场景 分段 模型查看预测样本**")
    c1, c2, c3 = st.columns(3)
    s = c1.selectbox("Scenario", sorted(preds["scenario"].unique()))
    g = c2.selectbox("Segment", sorted(preds["segment"].unique()))
    m = c3.selectbox("Model", sorted(preds["model"].unique()))

    view = preds[(preds["scenario"] == s) & (preds["segment"] == g) & (preds["model"] == m)]
    st.dataframe(view.head(300), use_container_width=True)

    st.markdown("**误差汇总 站点粒度**")
    if not errs.empty:
        st.dataframe(errs.head(200), use_container_width=True)



def show_explanation_text(data: Dict[str, object]) -> None:
    st.subheader("文字解释与结论")
    render_story_header(
        "上一页给出模型边界，这一步形成整体总结。",
        "当前流程已具备复跑能力，并验证了空间与事件信息对预测改进的价值。",
        "可在此基础上扩展更细粒度事件表与更强时空模型，持续提升稳定性。",
    )

    st.markdown("**结论草稿 `conclusion_draft.md`**")
    conclusion = data["conclusion"]
    if conclusion:
        st.markdown(conclusion)
    else:
        st.info("当前版本未找到 `conclusion_draft.md`")

    st.markdown("**报告摘要 `report.md`**")
    report = data["report"]
    if report:
        st.code(report[:8000], language="markdown")
    else:
        st.info("当前版本未找到 `report.md`")



def main() -> None:
    st.set_page_config(page_title="Casa0004 可视化看板", layout="wide")
    st.title("Casa0004 Pipeline 可视化看板")

    runs = list_runs()
    if not runs:
        st.error(f"未找到 run 目录 {OUTPUTS_DIR}")
        return

    run_names = [r.name for r in runs]
    default_idx = len(runs) - 1

    st.sidebar.header("运行选择")
    selected_name = st.sidebar.selectbox("结果版本", run_names, index=default_idx)
    selected_run = OUTPUTS_DIR / selected_name

    st.sidebar.markdown("---")
    st.sidebar.markdown("**阅读说明**")
    st.sidebar.markdown("- 页面按背景 方法 结果 解释 误差 结论顺序组织")
    st.sidebar.markdown("- 旧版 run 若缺少 `ablation` 或 `spatial` 文件会自动降级显示")
    st.sidebar.markdown("- 指标比较以 `R²` 为主，同时可结合 MAE 和 RMSE 观察")

    data = load_run_artifacts(selected_run)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["概览", "流程工件", "结果消融", "特征解释", "预测误差", "文字结论"]
    )

    with tab1:
        show_overview(selected_run, data)
    with tab2:
        show_pipeline_process(data)
    with tab3:
        show_metrics_and_ablation(data)
    with tab4:
        show_feature_insights(data)
    with tab5:
        show_predictions_and_errors(data)
    with tab6:
        show_explanation_text(data)


if __name__ == "__main__":
    main()
