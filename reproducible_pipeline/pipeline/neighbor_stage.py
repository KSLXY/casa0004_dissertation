from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors



def build_station_neighbors(
    merged: pd.DataFrame,
    station_col: str,
    lat_col: str,
    lon_col: str,
    k_neighbors: int,
    metric: str,
    output_dir: Path,
) -> pd.DataFrame:
    stations = (
        merged.groupby(station_col)
        .agg(lat=(lat_col, "first"), lon=(lon_col, "first"))
        .reset_index()
        .sort_values(station_col)
        .reset_index(drop=True)
    )

    coords = stations[["lat", "lon"]].to_numpy()
    knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(stations)), metric=metric)
    knn.fit(coords)
    distances, indices = knn.kneighbors(coords)

    rows = []
    for i, sid in enumerate(stations[station_col].values):
        rank = 0
        for dist, j in zip(distances[i], indices[i]):
            nid = stations.iloc[j][station_col]
            if nid == sid:
                continue
            rank += 1
            rows.append(
                {
                    "station_id": sid,
                    "neighbor_station_id": int(nid),
                    "rank": rank,
                    "distance_km": float(dist * 111.0),
                }
            )
            if rank >= k_neighbors:
                break

    neighbor_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    neighbor_df.to_csv(output_dir / "station_neighbors.csv", index=False)
    return neighbor_df



def add_neighbor_lag_features(
    df: pd.DataFrame,
    station_col: str,
    time_col: str,
    neighbor_df: pd.DataFrame,
    lags: list[int],
) -> pd.DataFrame:
    out = df.copy()

    station_ids = np.array(sorted(out[station_col].unique()))
    station_to_idx = {sid: i for i, sid in enumerate(station_ids)}

    W = np.zeros((len(station_ids), len(station_ids)), dtype=float)
    for r in neighbor_df.itertuples(index=False):
        i = station_to_idx.get(getattr(r, "station_id"))
        j = station_to_idx.get(getattr(r, "neighbor_station_id"))
        if i is None or j is None:
            continue
        # inverse-distance style weight using rank for stability
        W[i, j] = 1.0 / max(1, getattr(r, "rank"))

    row_sums = W.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    Wn = W / row_sums[:, None]

    for lag in lags:
        lag_col = f"lag_{lag}h_total_count"
        if lag_col not in out.columns:
            continue

        pivot = (
            out.pivot_table(index=time_col, columns=station_col, values=lag_col, aggfunc="mean")
            .reindex(columns=station_ids)
            .fillna(0.0)
        )

        X = pivot.to_numpy(dtype=float)
        neigh_mean = X @ Wn.T
        neigh_sum = X @ W.T

        mean_df = pd.DataFrame(neigh_mean, index=pivot.index, columns=station_ids).stack().rename(f"neighbor_mean_lag_{lag}h").reset_index()
        sum_df = pd.DataFrame(neigh_sum, index=pivot.index, columns=station_ids).stack().rename(f"neighbor_sum_lag_{lag}h").reset_index()
        mean_df.columns = [time_col, station_col, f"neighbor_mean_lag_{lag}h"]
        sum_df.columns = [time_col, station_col, f"neighbor_sum_lag_{lag}h"]

        out = out.merge(mean_df, on=[time_col, station_col], how="left")
        out = out.merge(sum_df, on=[time_col, station_col], how="left")

    if "neighbor_mean_lag_24h" in out.columns and "neighbor_sum_lag_24h" in out.columns:
        denom = out["neighbor_sum_lag_24h"].replace(0, np.nan)
        out["neighbor_peak_ratio_lag_24h"] = (out["neighbor_mean_lag_24h"] / denom).fillna(0.0)

    return out
