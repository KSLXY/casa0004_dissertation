from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler



def run_station_clustering(
    merged: pd.DataFrame,
    station_col: str,
    lat_col: str,
    lon_col: str,
    target_col: str,
    k_min: int,
    k_max: int,
    random_state: int,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    station_profile = (
        merged.groupby(station_col)
        .agg(
            lat=(lat_col, "first"),
            lon=(lon_col, "first"),
            demand_mean=(target_col, "mean"),
            demand_std=(target_col, "std"),
            demand_p95=(target_col, lambda x: np.percentile(x, 95)),
        )
        .reset_index()
        .fillna(0.0)
    )

    feats = station_profile[["lat", "lon", "demand_mean", "demand_std", "demand_p95"]].to_numpy()
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)

    best = None
    records = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        records.append({"k": k, "silhouette": float(sil)})
        if best is None or sil > best["silhouette"]:
            best = {
                "k": k,
                "silhouette": float(sil),
                "model": km,
                "labels": labels,
            }

    station_profile["cluster_id"] = best["labels"]

    summary = (
        station_profile.groupby("cluster_id")
        .agg(
            station_count=(station_col, "nunique"),
            mean_demand=("demand_mean", "mean"),
            center_lat=("lat", "mean"),
            center_lon=("lon", "mean"),
        )
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = output_dir / "cluster_artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    with (artifacts / "cluster_model.pkl").open("wb") as f:
        pickle.dump({"model": best["model"], "scaler": scaler, "feature_cols": ["lat", "lon", "demand_mean", "demand_std", "demand_p95"]}, f)

    station_profile[[station_col, "cluster_id"]].to_csv(artifacts / "station_cluster_map.csv", index=False)
    summary.to_csv(artifacts / "cluster_summary.csv", index=False)
    (artifacts / "k_search_metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    meta = {
        "best_k": int(best["k"]),
        "best_silhouette": float(best["silhouette"]),
        "k_search": records,
    }
    return station_profile[[station_col, "cluster_id"]], meta



def add_cluster_lag_features(
    df: pd.DataFrame,
    station_cluster_map: pd.DataFrame,
    station_col: str,
    cluster_col: str,
    time_col: str,
    target_col: str,
    lags: list[int],
) -> pd.DataFrame:
    out = df.merge(station_cluster_map, on=station_col, how="left")
    out = out.rename(columns={"cluster_id": cluster_col})

    cluster_ts = (
        out.groupby([cluster_col, time_col])[target_col]
        .mean()
        .reset_index(name="cluster_demand")
    )

    for lag in lags:
        lagged = cluster_ts.copy()
        lagged[time_col] = lagged[time_col] + pd.to_timedelta(lag, unit="h")
        out = out.merge(
            lagged.rename(columns={"cluster_demand": f"cluster_demand_lag_{lag}h"}),
            on=[cluster_col, time_col],
            how="left",
        )

    return out
