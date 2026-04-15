from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd



def _daily_event_frame(calendar_csv: Path, start_date: str, end_date: str) -> pd.DataFrame:
    cal = pd.read_csv(calendar_csv)
    cal["date"] = pd.to_datetime(cal["date"]).dt.date

    days = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D").date})

    cal = cal.sort_values(["date", "event_weight"], ascending=[True, False])
    agg = (
        cal.groupby("date")
        .agg(
            is_public_holiday=("event_type", lambda x: int((x == "public_holiday").any())),
            is_school_break=("event_type", lambda x: int((x == "school_break").any())),
            is_major_event=("event_type", lambda x: int((x == "major_event").any())),
            event_weight=("event_weight", "max"),
        )
        .reset_index()
    )

    out = days.merge(agg, on="date", how="left")
    for c in ["is_public_holiday", "is_school_break", "is_major_event"]:
        out[c] = out[c].fillna(0).astype(int)
    out["event_weight"] = out["event_weight"].fillna(0.0)

    event_days = out.loc[out["event_weight"] > 0, "date"].tolist()
    if not event_days:
        out["days_since_prev_event"] = 999
        out["days_to_next_event"] = 999
        return out

    evt_np = np.array(pd.to_datetime(event_days).astype("int64"))
    day_np = np.array(pd.to_datetime(out["date"]).astype("int64"))

    prev_vals = []
    next_vals = []
    for d in day_np:
        prev = evt_np[evt_np <= d]
        nxt = evt_np[evt_np >= d]
        prev_vals.append(int((d - prev.max()) / 86_400_000_000_000) if len(prev) else 999)
        next_vals.append(int((nxt.min() - d) / 86_400_000_000_000) if len(nxt) else 999)

    out["days_since_prev_event"] = prev_vals
    out["days_to_next_event"] = next_vals
    return out



def add_event_features(df: pd.DataFrame, time_col: str, calendar_csv: Path) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out[time_col]).dt.date
    start = str(pd.to_datetime(out[time_col]).min().date())
    end = str(pd.to_datetime(out[time_col]).max().date())

    daily = _daily_event_frame(calendar_csv, start, end)
    out = out.merge(daily, on="date", how="left")

    for c in ["is_public_holiday", "is_school_break", "is_major_event", "days_since_prev_event", "days_to_next_event"]:
        out[c] = out[c].fillna(0)
    out["event_weight"] = out["event_weight"].fillna(0.0)
    return out
