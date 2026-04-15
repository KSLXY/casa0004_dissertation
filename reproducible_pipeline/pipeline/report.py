from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def build_error_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    grp = (
        predictions.groupby(["scenario", "segment", "model", "station_id"], dropna=False)
        .agg(
            n=("y_true", "size"),
            mae=("abs_residual", "mean"),
            rmse=("residual", lambda x: (x.pow(2).mean()) ** 0.5),
            mean_true=("y_true", "mean"),
            mean_pred=("y_pred", "mean"),
        )
        .reset_index()
        .sort_values(["scenario", "segment", "model", "mae"], ascending=[True, True, True, False])
    )
    return grp



def build_ablation_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.sort_values(["scenario", "segment", "R2"], ascending=[True, True, False])
        .groupby(["scenario", "segment"], as_index=False)
        .head(1)
    )



def build_spatial_feature_importance(feature_importance: pd.DataFrame) -> pd.DataFrame:
    if feature_importance.empty:
        return feature_importance.copy()
    keys = [
        "cluster",
        "neighbor",
        "event",
        "holiday",
        "school_break",
        "major_event",
        "days_since_prev_event",
        "days_to_next_event",
    ]
    mask = feature_importance["feature"].str.contains("|".join(keys), case=False, na=False)
    return feature_importance[mask].sort_values(
        ["scenario", "segment", "importance"], ascending=[True, True, False]
    )



def write_markdown_summary(output_dir: Path, metrics: pd.DataFrame, data_version: dict, config_text: str) -> None:
    lines: list[str] = []
    lines.append("# Casa0004 Reproducible Pipeline Report")
    lines.append("")
    lines.append("## 1. Data Version")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(data_version, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## 2. Metrics Summary (Full Scenario)")
    lines.append("")

    full = metrics[metrics["scenario"] == "full"]
    best = full.sort_values(["segment", "R2"], ascending=[True, False]).groupby("segment", as_index=False).head(1)
    for _, r in best.iterrows():
        lines.append(
            f"- `{r['segment']}` best model: `{r['model']}` | R2={r['R2']:.4f}, MAE={r['MAE']:.4f}, RMSE={r['RMSE']:.4f}, WAPE={r['WAPE']:.2f}%"
        )

    lines.append("")
    lines.append("## 3. Full Configuration Snapshot")
    lines.append("")
    lines.append("```toml")
    lines.append(config_text)
    lines.append("```")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")



def write_key_plots(output_dir: Path, metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    plot_df = metrics[metrics["scenario"] == "full"].copy()
    plot_df["segment_model"] = plot_df["segment"] + "-" + plot_df["model"]

    plt.figure(figsize=(10, 5))
    ordered = plot_df.sort_values("R2", ascending=False)
    plt.bar(ordered["segment_model"], ordered["R2"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("R2")
    plt.title("Model Comparison by Segment (Full)")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sample = predictions[predictions["scenario"] == "full"].copy()
    sample["segment_model"] = sample["segment"] + "-" + sample["model"]
    sample = sample[sample["segment_model"].isin(ordered["segment_model"].head(4))]
    grouped = [
        sample[sample["segment_model"] == name]["residual"].values
        for name in ordered["segment_model"].head(4)
    ]
    plt.boxplot(grouped, tick_labels=list(ordered["segment_model"].head(4)), showfliers=False)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Residual")
    plt.title("Residual Distribution (Top 4 Segment-Models, Full)")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_distribution.png", dpi=160)
    plt.close()



def write_conclusion_draft(output_dir: Path, metrics: pd.DataFrame, baseline_metrics: pd.DataFrame | None = None) -> None:
    full = metrics[metrics["scenario"] == "full"]
    best = full.sort_values(["segment", "R2"], ascending=[True, False]).groupby("segment", as_index=False).head(1)
    rows = {r["segment"]: r for _, r in best.iterrows()}
    wkd = rows.get("weekday")
    wnd = rows.get("weekend")

    text = [
        "# Conclusion Draft",
        "",
        "This enhancement round integrates re-clustering, neighbor flow spillover, and a full event calendar.",
        (
            f"For weekday demand, the best full-scenario model is `{wkd['model']}` (R2={wkd['R2']:.3f}, "
            f"MAE={wkd['MAE']:.3f}, RMSE={wkd['RMSE']:.3f})."
            if wkd is not None
            else "Weekday best-model summary is unavailable."
        ),
        (
            f"For weekend demand, the best full-scenario model is `{wnd['model']}` (R2={wnd['R2']:.3f}, "
            f"MAE={wnd['MAE']:.3f}, RMSE={wnd['RMSE']:.3f})."
            if wnd is not None
            else "Weekend best-model summary is unavailable."
        ),
        "Ablation outputs quantify the incremental contribution of cluster, neighbor, and event features.",
    ]

    if baseline_metrics is not None and not baseline_metrics.empty:
        b = baseline_metrics.sort_values(["segment", "R2"], ascending=[True, False]).groupby("segment", as_index=False).head(1)
        bmap = {r["segment"]: r for _, r in b.iterrows()}
        lines = []
        for seg, cur in [("weekday", wkd), ("weekend", wnd)]:
            if cur is None or seg not in bmap:
                continue
            delta = cur["R2"] - bmap[seg]["R2"]
            lines.append(f"- `{seg}` R2 change vs baseline: {delta:+.4f}")
        if lines:
            text.extend(["", "## Improvement vs run_20260415_102043", *lines])

    (output_dir / "conclusion_draft.md").write_text("\n".join(text), encoding="utf-8")
