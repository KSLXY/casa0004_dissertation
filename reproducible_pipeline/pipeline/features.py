from __future__ import annotations

import pandas as pd
import numpy as np



def build_feature_frame(
    merged: pd.DataFrame,
    target_col: str,
    station_col: str,
    time_col: str,
    lag_hours: list[int],
    add_lag_features: bool,
) -> pd.DataFrame:
    df = merged.copy()

    df = df.sort_values([station_col, time_col]).reset_index(drop=True)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df["date"] = df[time_col].dt.date
    df["month"] = df[time_col].dt.month
    df["day_of_week"] = df[time_col].dt.dayofweek
    df["is_peak"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    if add_lag_features:
        for lag in lag_hours:
            df[f"lag_{lag}h_{target_col}"] = (
                df.groupby(station_col)[target_col].shift(lag)
            )

    df = df.dropna(subset=[time_col, target_col])
    return df



def split_weekday_weekend(df: pd.DataFrame, weekend_col: str = "is_weekend") -> tuple[pd.DataFrame, pd.DataFrame]:
    weekday_df = df[df[weekend_col] == 0].copy()
    weekend_df = df[df[weekend_col] == 1].copy()
    return weekday_df, weekend_df



def time_split(df: pd.DataFrame, time_col: str, train_end: str, val_end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_ts = pd.Timestamp(train_end)
    val_end_ts = pd.Timestamp(val_end)

    train = df[df[time_col] < train_end_ts].copy()
    val = df[(df[time_col] >= train_end_ts) & (df[time_col] < val_end_ts)].copy()
    test = df[df[time_col] >= val_end_ts].copy()

    return train, val, test
