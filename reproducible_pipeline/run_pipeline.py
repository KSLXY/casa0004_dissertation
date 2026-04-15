from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json

import pandas as pd

from pipeline.cluster_stage import add_cluster_lag_features, run_station_clustering
from pipeline.config import load_config
from pipeline.data_ingest import load_inputs, save_data_version
from pipeline.event_stage import add_event_features
from pipeline.features import build_feature_frame, split_weekday_weekend, time_split
from pipeline.neighbor_stage import add_neighbor_lag_features, build_station_neighbors
from pipeline.report import (
    build_ablation_summary,
    build_error_analysis,
    build_spatial_feature_importance,
    write_conclusion_draft,
    write_key_plots,
    write_markdown_summary,
)
from pipeline.train import train_and_evaluate



def _resolve_run_dir(base_dir: Path, run_name: str) -> Path:
    if run_name == "auto":
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir



def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def _scenario_feature_map(all_cols: list[str]) -> dict[str, list[str]]:
    cluster_cols = [c for c in all_cols if c.startswith("cluster_") or c == "cluster_id"]
    neighbor_cols = [c for c in all_cols if c.startswith("neighbor_")]
    event_cols = [
        c
        for c in all_cols
        if c
        in {
            "is_public_holiday",
            "is_school_break",
            "is_major_event",
            "event_weight",
            "days_to_next_event",
            "days_since_prev_event",
        }
    ]

    base_cols = [c for c in all_cols if c not in set(cluster_cols + neighbor_cols + event_cols)]

    return {
        "baseline_current_features": base_cols,
        "cluster_only": base_cols + cluster_cols,
        "neighbor_only": base_cols + neighbor_cols,
        "events_only": base_cols + event_cols,
        "full": base_cols + cluster_cols + neighbor_cols + event_cols,
    }



def _load_baseline_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    cols = {"segment", "model", "R2", "MAE", "RMSE", "WAPE"}
    if not cols.issubset(df.columns):
        return pd.DataFrame()
    return df



def main() -> None:
    parser = argparse.ArgumentParser(description="Casa0004 reproducible pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("reproducible_pipeline/config/default.toml"),
        help="Path to TOML config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = _resolve_run_dir(cfg.output.output_dir, cfg.output.run_name)

    config_text = args.config.read_text(encoding="utf-8")
    (run_dir / "config_snapshot.toml").write_text(config_text, encoding="utf-8")

    weather, merged, dockless, data_version = load_inputs(
        cfg.data.weather_csv,
        cfg.data.merged_station_hour_csv,
        cfg.data.dockless_trip_csv,
    )
    save_data_version(data_version, run_dir)

    # 1) Base temporal/weather feature frame
    frame = build_feature_frame(
        merged=merged,
        target_col=cfg.feature.target_col,
        station_col=cfg.feature.station_id_col,
        time_col=cfg.feature.time_col,
        lag_hours=cfg.feature.lag_hours,
        add_lag_features=cfg.feature.add_lag_features,
    )

    # 2) Cluster stage
    station_cluster_map, cluster_meta = run_station_clustering(
        merged=merged,
        station_col=cfg.feature.station_id_col,
        lat_col="lat",
        lon_col="lon",
        target_col=cfg.feature.target_col,
        k_min=cfg.cluster.k_min,
        k_max=cfg.cluster.k_max,
        random_state=cfg.cluster.random_state,
        output_dir=run_dir,
    )
    frame = add_cluster_lag_features(
        df=frame,
        station_cluster_map=station_cluster_map,
        station_col=cfg.feature.station_id_col,
        cluster_col="cluster_id",
        time_col=cfg.feature.time_col,
        target_col=cfg.feature.target_col,
        lags=[1, 24],
    )

    # 3) Neighbor stage
    neighbors = build_station_neighbors(
        merged=merged,
        station_col=cfg.feature.station_id_col,
        lat_col="lat",
        lon_col="lon",
        k_neighbors=cfg.neighbor.k_neighbors,
        metric=cfg.neighbor.distance_metric,
        output_dir=run_dir,
    )
    frame = add_neighbor_lag_features(
        df=frame,
        station_col=cfg.feature.station_id_col,
        time_col=cfg.feature.time_col,
        neighbor_df=neighbors,
        lags=[1, 24],
    )

    # 4) Event stage
    if cfg.feature.use_event_features:
        frame = add_event_features(
            df=frame,
            time_col=cfg.feature.time_col,
            calendar_csv=cfg.feature.events_calendar_csv,
        )

    weekday_df, weekend_df = split_weekday_weekend(frame, weekend_col="is_weekend")

    # Save the full engineered frame once for schema auditing
    _save_csv(frame.head(20000), run_dir / "feature_frame_sample.csv")

    all_metrics = []
    all_predictions = []
    all_importance = []

    all_cols = [c for c in frame.columns if c not in {cfg.feature.target_col, cfg.feature.time_col, "date"}]
    scenarios = _scenario_feature_map(all_cols)

    for scenario_name, selected_features in scenarios.items():
        for segment_name, segment_df in [("weekday", weekday_df), ("weekend", weekend_df)]:
            train_df, val_df, test_df = time_split(
                segment_df,
                time_col=cfg.feature.time_col,
                train_end=cfg.split.train_end,
                val_end=cfg.split.val_end,
            )

            if scenario_name == "full":
                _save_csv(train_df, run_dir / f"{segment_name}_train_split.csv")
                _save_csv(val_df, run_dir / f"{segment_name}_val_split.csv")
                _save_csv(test_df, run_dir / f"{segment_name}_test_split.csv")

            result = train_and_evaluate(
                train_df=train_df,
                test_df=test_df,
                target_col=cfg.feature.target_col,
                time_col=cfg.feature.time_col,
                random_state=cfg.model.random_state,
                rf_n_estimators=cfg.model.random_forest_n_estimators,
                rf_max_depth=cfg.model.random_forest_max_depth,
                rf_min_samples_leaf=cfg.model.random_forest_min_samples_leaf,
                segment_name=segment_name,
                scenario_name=scenario_name,
                selected_features=selected_features,
            )

            all_metrics.append(result.metrics)
            all_predictions.append(result.predictions)
            all_importance.append(result.feature_importance)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    feature_importance_df = pd.concat(all_importance, ignore_index=True)
    error_analysis_df = build_error_analysis(predictions_df)
    ablation_df = build_ablation_summary(metrics_df)
    spatial_fi_df = build_spatial_feature_importance(feature_importance_df)

    _save_csv(metrics_df, run_dir / "metrics_all.csv")
    _save_csv(ablation_df, run_dir / "ablation_metrics.csv")
    _save_csv(feature_importance_df, run_dir / "feature_importance.csv")
    _save_csv(spatial_fi_df, run_dir / "spatial_feature_importance.csv")
    _save_csv(predictions_df, run_dir / "predictions.csv")
    _save_csv(error_analysis_df, run_dir / "error_analysis.csv")

    baseline_df = _load_baseline_metrics(cfg.evaluation.baseline_metrics_csv)

    write_key_plots(run_dir, metrics_df, predictions_df)
    write_conclusion_draft(run_dir, metrics_df, baseline_metrics=baseline_df)
    write_markdown_summary(run_dir, metrics_df, data_version, config_text)

    full_best = (
        metrics_df[metrics_df["scenario"] == "full"]
        .sort_values(["segment", "R2"], ascending=[True, False])
        .groupby("segment", as_index=False)
        .head(1)
    )

    baseline_best = (
        baseline_df.sort_values(["segment", "R2"], ascending=[True, False]).groupby("segment", as_index=False).head(1)
        if not baseline_df.empty
        else pd.DataFrame(columns=["segment", "R2"])
    )

    delta_records = []
    if not baseline_best.empty:
        merged_delta = full_best[["segment", "R2"]].merge(
            baseline_best[["segment", "R2"]],
            on="segment",
            suffixes=("_full", "_baseline"),
            how="left",
        )
        merged_delta["delta_R2"] = merged_delta["R2_full"] - merged_delta["R2_baseline"]
        delta_records = merged_delta.to_dict(orient="records")

    summary = {
        "created_at": datetime.now().isoformat(),
        "rows": {
            "feature_frame": int(len(frame)),
            "predictions": int(len(predictions_df)),
        },
        "cluster": cluster_meta,
        "best_full_by_segment": full_best[["segment", "model", "R2", "MAE", "RMSE", "WAPE"]].to_dict(orient="records"),
        "delta_vs_baseline": delta_records,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Run completed. Artifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
