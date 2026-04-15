from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT / "outputs"


I18N = {
    "zh": {
        "title": "Casa0004 共享单车故事看板",
        "lang": "语言 / Language",
        "run_select": "查看哪个版本的结果",
        "current_run": "结果版本",
        "note": "阅读说明",
        "note1": "这是一条连续故事线：背景 → 方法 → 结果 → 原因 → 边界 → 下一步。",
        "note2": "专业术语被放在“补充解释”里，主文案尽量通俗。",
        "tabs": ["1 背景", "2 方法", "3 结果", "4 原因", "5 边界", "6 下一步"],
        "story_steps": ["背景问题", "如何做", "做出来什么", "为什么有效", "风险与不足", "改进空间"],
        "header_q": "这一步回答的问题",
        "header_prev": "承接上一页",
        "header_core": "本页核心结论",
        "header_next": "看完后建议去",
        "continue": "继续阅读",
        "continue_hint": "请点击上方 Tab：",
        "missing": "当前版本缺少这个文件，已自动降级显示。",
        "overview_intro": "如果你是第一次看，请把这个项目理解为：我们想提前知道哪里、什么时候会更需要共享单车，这样调度更省力，用户更容易借到车。",
        "overview_bg": "都柏林共享单车同时受到通勤规律、天气、节假日和空间位置影响，需求会明显波动。",
        "overview_goal": "目标是把这些信息汇总成一个可重复运行的预测流程，并解释模型为什么这么判断。",
        "rows_feature": "特征行数",
        "rows_pred": "预测行数",
        "summary_json": "本次运行摘要",
        "method_intro": "这一页讲‘怎么做’：我们不是直接训练模型，而是先把原始数据一步步变成有意义的特征。",
        "method_block1": "数据接入：校验字段是否齐全，记录数据版本，保证复现。",
        "method_block2": "空间增强：给站点分群（cluster）并找邻近站点，让模型看到‘周边影响’。",
        "method_block3": "事件增强：把节假日、学校假期、大型活动加入同一天的特征。",
        "method_block4": "训练设计：工作日和周末分开训练，并做 5 组消融实验看每个改动是否真的有用。",
        "cluster_summary": "站点分群摘要",
        "neighbor_top": "站点邻接示例（前20行）",
        "result_intro": "这一页看‘做出来什么’：核心看 R²（越高代表模型越能解释真实波动）。",
        "metrics_table": "完整指标表（可筛选）",
        "ablation_best": "每个场景的最佳结果（Ablation）",
        "result_takeaway": "如果 full 场景的 R² 高于其他场景，说明“空间+事件”这套增强是有价值的。",
        "feature_intro": "这一页解释‘为什么有效’：模型最依赖哪些信息来做判断。",
        "spatial_fi": "空间/事件特征重要性",
        "full_fi": "查看完整特征重要性表",
        "feature_takeaway": "如果 cluster/neighbor/event 相关特征排名靠前，说明它们确实在帮模型抓住规律。",
        "pred_intro": "这一页看‘哪里稳、哪里不稳’：用残差图和样本表观察误差模式。",
        "resid_img": "残差分布图",
        "model_img": "模型对比图",
        "view_pred": "按场景/分段/模型查看预测样本",
        "err_station": "站点级误差汇总",
        "pred_takeaway": "误差大的站点通常代表异常波动多，后续需要补更多上下文特征。",
        "next_intro": "最后一页把前面内容收束成结论，并给出下一步可落地优化。",
        "conclusion": "结论草稿",
        "report": "报告摘要",
        "next_actions": "建议下一步",
        "action1": "补充更细粒度事件数据（演唱会、比赛、施工影响）。",
        "action2": "引入实时数据更新机制，做滚动预测。",
        "action3": "增加站点运营约束（车辆上限、调度成本）做业务优化。",
        "glossary": "补充解释（术语）",
        "g_r2": "R²：衡量模型解释能力的指标，通常越高越好。",
        "g_ablation": "消融（Ablation）：每次只增加一个改动，比较是否真的带来提升。",
        "g_fi": "特征重要性：模型在预测时更依赖哪些输入信息。",
    },
    "en": {
        "title": "Casa0004 Bike-Sharing Story Dashboard",
        "lang": "Language / 语言",
        "run_select": "Choose result version",
        "current_run": "Result Version",
        "note": "How to read",
        "note1": "This dashboard is a continuous story: problem -> method -> result -> reason -> boundary -> next step.",
        "note2": "Technical terms are moved into “Glossary”, while the main text stays plain-language.",
        "tabs": ["1 Background", "2 Method", "3 Results", "4 Why It Works", "5 Limits", "6 Next Step"],
        "story_steps": ["Problem", "How we did it", "What we got", "Why it works", "Risks & limits", "Improvement"],
        "header_q": "Question answered in this step",
        "header_prev": "Bridge from previous tab",
        "header_core": "Core takeaway",
        "header_next": "Where to go next",
        "continue": "Continue",
        "continue_hint": "Please open the tab:",
        "missing": "This file is missing in the selected run. Fallback view is shown.",
        "overview_intro": "If this is your first time here: the goal is to predict where/when shared bikes are needed, so operations can move bikes earlier and users can find bikes more easily.",
        "overview_bg": "Bike demand in Dublin changes with commuting patterns, weather, holidays, and location context.",
        "overview_goal": "We build a reproducible pipeline and explain not only “how accurate” but also “why the model decides this way.”",
        "rows_feature": "Feature Rows",
        "rows_pred": "Prediction Rows",
        "summary_json": "Run summary",
        "method_intro": "This tab explains the method: we do not train directly on raw data; we first transform data into meaningful signals.",
        "method_block1": "Ingest: validate schema and track data version for reproducibility.",
        "method_block2": "Spatial enhancement: cluster stations and add neighbor spillover signals.",
        "method_block3": "Event enhancement: merge public holidays, school breaks, and major events.",
        "method_block4": "Training design: split weekday/weekend and run 5-scenario ablation to verify incremental value.",
        "cluster_summary": "Cluster summary",
        "neighbor_top": "Neighbor sample (top 20)",
        "result_intro": "This tab shows what we got. Main metric is R² (higher usually means better explanatory power).",
        "metrics_table": "Full metrics table (filterable)",
        "ablation_best": "Best result by scenario (ablation)",
        "result_takeaway": "If the full scenario beats others on R², the spatial+event enhancements are likely useful.",
        "feature_intro": "This tab explains why it works: which signals the model relies on most.",
        "spatial_fi": "Spatial/Event feature importance",
        "full_fi": "Show full feature importance table",
        "feature_takeaway": "If cluster/neighbor/event features rank high, they are likely driving performance gains.",
        "pred_intro": "This tab shows where the model is stable or unstable using residuals and sample predictions.",
        "resid_img": "Residual distribution",
        "model_img": "Model comparison",
        "view_pred": "View prediction samples by scenario/segment/model",
        "err_station": "Station-level error summary",
        "pred_takeaway": "Large-error stations often indicate stronger external shocks; they need richer context features.",
        "next_intro": "This final tab wraps up the story and gives actionable next improvements.",
        "conclusion": "Conclusion draft",
        "report": "Report snapshot",
        "next_actions": "Suggested next actions",
        "action1": "Add more fine-grained events (concerts, sports, roadworks).",
        "action2": "Introduce rolling updates for near-real-time forecasting.",
        "action3": "Add operation constraints (station capacity, dispatch cost) for business optimization.",
        "glossary": "Glossary",
        "g_r2": "R²: model explanatory score; higher is usually better.",
        "g_ablation": "Ablation: add one change at a time to verify real incremental value.",
        "g_fi": "Feature importance: which inputs the model depends on most.",
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


def render_story_nav(current_idx: int, t: Dict[str, object]) -> None:
    labels = t["story_steps"]
    cols = st.columns(6)
    for i, label in enumerate(labels):
        if i == current_idx:
            cols[i].markdown(f"**➡ {i+1}. {label}**")
        else:
            cols[i].markdown(f"{i+1}. {label}")


def render_step_header(current_idx: int, prev_text: str, core_text: str, next_text: str, t: Dict[str, object]) -> None:
    render_story_nav(current_idx, t)
    st.markdown("---")
    st.markdown(f"**{t['header_q']}**: {t['story_steps'][current_idx]}")
    st.markdown(f"**{t['header_prev']}**: {prev_text}")
    st.markdown(f"**{t['header_core']}**: {core_text}")
    st.markdown(f"**{t['header_next']}**: {next_text}")


def render_continue_hint(next_tab_label: str, t: Dict[str, object], key: str) -> None:
    if st.button(t["continue"], key=key):
        st.info(f"{t['continue_hint']} {next_tab_label}")


def show_overview(data: Dict[str, object], current_run: Path, t: Dict[str, object]) -> None:
    render_step_header(
        0,
        prev_text="这是故事起点。" if "背景" in t["story_steps"][0] else "Story starts here.",
        core_text=t["overview_goal"],
        next_text=t["tabs"][1],
        t=t,
    )
    st.markdown(f"- {t['overview_intro']}")
    st.markdown(f"- {t['overview_bg']}")

    summary = data["summary"]
    c1, c2, c3 = st.columns(3)
    c1.metric(t["current_run"], current_run.name)
    c2.metric(t["rows_feature"], f"{summary.get('rows', {}).get('feature_frame', 'N/A')}")
    c3.metric(t["rows_pred"], f"{summary.get('rows', {}).get('predictions', 'N/A')}")

    st.markdown(f"**{t['summary_json']}**")
    st.json(summary if summary else {})

    with st.expander(t["glossary"]):
        st.markdown(f"- {t['g_r2']}")
        st.markdown(f"- {t['g_ablation']}")
        st.markdown(f"- {t['g_fi']}")

    render_continue_hint(t["tabs"][1], t, key="continue_0")


def show_method(data: Dict[str, object], t: Dict[str, object]) -> None:
    render_step_header(1, t["overview_goal"], t["method_intro"], t["tabs"][2], t)
    st.markdown(f"1. {t['method_block1']}")
    st.markdown(f"2. {t['method_block2']}")
    st.markdown(f"3. {t['method_block3']}")
    st.markdown(f"4. {t['method_block4']}")

    c1, c2 = st.columns(2)
    c1.markdown(f"**{t['cluster_summary']}**")
    if data["cluster_summary"].empty:
        c1.info(t["missing"])
    else:
        c1.dataframe(data["cluster_summary"], use_container_width=True)

    c2.markdown(f"**{t['neighbor_top']}**")
    if data["neighbors"].empty:
        c2.info(t["missing"])
    else:
        c2.dataframe(data["neighbors"].head(20), use_container_width=True)

    render_continue_hint(t["tabs"][2], t, key="continue_1")


def show_results(data: Dict[str, object], t: Dict[str, object]) -> None:
    render_step_header(2, t["method_intro"], t["result_intro"], t["tabs"][3], t)

    metrics = ensure_scenario(data["metrics"])
    if metrics.empty:
        st.warning(t["missing"])
        return

    scenarios = sorted(metrics["scenario"].unique())
    segments = sorted(metrics["segment"].unique())
    models = sorted(metrics["model"].unique())

    if "filter_scenarios" not in st.session_state:
        st.session_state["filter_scenarios"] = scenarios
    if "filter_segments" not in st.session_state:
        st.session_state["filter_segments"] = segments
    if "filter_models" not in st.session_state:
        st.session_state["filter_models"] = models

    c1, c2, c3 = st.columns(3)
    st.session_state["filter_scenarios"] = c1.multiselect("Scenario", scenarios, default=st.session_state["filter_scenarios"])
    st.session_state["filter_segments"] = c2.multiselect("Segment", segments, default=st.session_state["filter_segments"])
    st.session_state["filter_models"] = c3.multiselect("Model", models, default=st.session_state["filter_models"])

    view = metrics[
        metrics["scenario"].isin(st.session_state["filter_scenarios"])
        & metrics["segment"].isin(st.session_state["filter_segments"])
        & metrics["model"].isin(st.session_state["filter_models"])
    ].sort_values(["scenario", "segment", "R2"], ascending=[True, True, False])

    st.markdown(f"**{t['metrics_table']}**")
    st.dataframe(view, use_container_width=True)

    ablation = data["ablation"]
    if ablation.empty:
        ablation = metrics.sort_values(["scenario", "segment", "R2"], ascending=[True, True, False]).groupby(["scenario", "segment"], as_index=False).head(1)

    st.markdown(f"**{t['ablation_best']}**")
    st.dataframe(ablation, use_container_width=True)

    chart = ablation[["scenario", "segment", "R2"]].copy()
    chart["name"] = chart["scenario"] + " | " + chart["segment"]
    st.bar_chart(chart.set_index("name")["R2"], height=320)

    st.success(t["result_takeaway"])
    render_continue_hint(t["tabs"][3], t, key="continue_2")


def show_feature_insights(data: Dict[str, object], t: Dict[str, object]) -> None:
    render_step_header(3, t["result_intro"], t["feature_intro"], t["tabs"][4], t)

    spatial = data["spatial_fi"]
    if spatial.empty:
        st.warning(t["missing"])
        return

    st.markdown(f"**{t['spatial_fi']}**")
    st.dataframe(spatial.head(100), use_container_width=True)

    focus = spatial.copy()
    if "scenario" in focus.columns and "full" in focus["scenario"].unique():
        focus = focus[focus["scenario"] == "full"]
    top = focus.sort_values("importance", ascending=False).head(20)
    top = top.assign(label=top["segment"] + " | " + top["feature"])
    st.bar_chart(top.set_index("label")["importance"], height=340)

    with st.expander(t["full_fi"]):
        full_fi = data["feature_importance"]
        if full_fi.empty:
            st.info(t["missing"])
        else:
            st.dataframe(full_fi.head(200), use_container_width=True)

    st.success(t["feature_takeaway"])
    render_continue_hint(t["tabs"][4], t, key="continue_3")


def show_predictions(data: Dict[str, object], t: Dict[str, object]) -> None:
    render_step_header(4, t["feature_intro"], t["pred_intro"], t["tabs"][5], t)

    preds = ensure_scenario(data["predictions"])
    errs = data["error_analysis"]
    if preds.empty:
        st.warning(t["missing"])
        return

    st.markdown(f"**{t['resid_img']}**")
    if data["residual_img"].exists():
        st.image(str(data["residual_img"]), use_container_width=True)
    else:
        st.info(t["missing"])

    st.markdown(f"**{t['model_img']}**")
    if data["model_comparison_img"].exists():
        st.image(str(data["model_comparison_img"]), use_container_width=True)
    else:
        st.info(t["missing"])

    st.markdown(f"**{t['view_pred']}**")
    c1, c2, c3 = st.columns(3)
    s = c1.selectbox("Scenario", sorted(preds["scenario"].unique()), key="pred_scenario")
    g = c2.selectbox("Segment", sorted(preds["segment"].unique()), key="pred_segment")
    m = c3.selectbox("Model", sorted(preds["model"].unique()), key="pred_model")

    pred_view = preds[(preds["scenario"] == s) & (preds["segment"] == g) & (preds["model"] == m)]
    st.dataframe(pred_view.head(300), use_container_width=True)

    st.markdown(f"**{t['err_station']}**")
    if errs.empty:
        st.info(t["missing"])
    else:
        st.dataframe(errs.head(200), use_container_width=True)

    st.success(t["pred_takeaway"])
    render_continue_hint(t["tabs"][5], t, key="continue_4")


def show_next_steps(data: Dict[str, object], t: Dict[str, object]) -> None:
    render_step_header(5, t["pred_intro"], t["next_intro"], "—", t)

    st.markdown(f"**{t['conclusion']}**")
    if data["conclusion"]:
        st.markdown(data["conclusion"])
    else:
        st.info(t["missing"])

    st.markdown(f"**{t['report']}**")
    if data["report"]:
        st.code(data["report"][:8000], language="markdown")
    else:
        st.info(t["missing"])

    st.markdown(f"**{t['next_actions']}**")
    st.markdown(f"1. {t['action1']}")
    st.markdown(f"2. {t['action2']}")
    st.markdown(f"3. {t['action3']}")


def main() -> None:
    st.set_page_config(page_title="Casa0004 Story Dashboard", layout="wide")

    lang = st.sidebar.radio(I18N["zh"]["lang"], ["中文", "English"], index=0)
    key = "zh" if lang == "中文" else "en"
    t = I18N[key]

    st.title(t["title"])

    runs = list_runs()
    if not runs:
        st.error(f"No runs found under: {OUTPUTS_DIR}")
        return

    run_names = [r.name for r in runs]
    selected_name = st.sidebar.selectbox(t["current_run"], run_names, index=len(run_names) - 1)
    selected_run = OUTPUTS_DIR / selected_name

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{t['run_select']}**")
    st.sidebar.markdown(f"`{selected_name}`")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{t['note']}**")
    st.sidebar.markdown(f"- {t['note1']}")
    st.sidebar.markdown(f"- {t['note2']}")

    data = load_run_artifacts(selected_run)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(t["tabs"])

    with tab1:
        show_overview(data, selected_run, t)
    with tab2:
        show_method(data, t)
    with tab3:
        show_results(data, t)
    with tab4:
        show_feature_insights(data, t)
    with tab5:
        show_predictions(data, t)
    with tab6:
        show_next_steps(data, t)


if __name__ == "__main__":
    main()
