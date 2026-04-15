from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json

import pandas as pd


REQUIRED_MERGED_COLS = {
    "station_id",
    "hourly",
    "rental_count",
    "dockless_count",
    "dockless_offnet",
    "total_count",
    "rain",
    "temp",
    "rhum",
    "wetb",
    "dewpt",
    "vappr",
    "msl",
    "hour",
    "is_weekend",
}

REQUIRED_WEATHER_COLS = {
    "datetime",
    "rain",
    "temp",
    "rhum",
    "wetb",
    "dewpt",
    "vappr",
    "msl",
    "hour",
}



def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        chunk = f.read(1024 * 1024)
        hasher.update(chunk)
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_time": stat.st_mtime,
        "sha256_head_1mb": hasher.hexdigest(),
    }



def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")



def load_inputs(weather_csv: Path, merged_csv: Path, dockless_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    weather = pd.read_csv(weather_csv)
    merged = pd.read_csv(merged_csv)
    dockless = pd.read_csv(dockless_csv)

    _validate_columns(weather, REQUIRED_WEATHER_COLS, "weather")
    _validate_columns(merged, REQUIRED_MERGED_COLS, "merged_station_hour")

    weather["datetime"] = pd.to_datetime(weather["datetime"], errors="coerce")
    merged["hourly"] = pd.to_datetime(merged["hourly"], errors="coerce")

    if "trip_start_time" in dockless.columns:
        dockless["trip_start_time"] = pd.to_datetime(dockless["trip_start_time"], errors="coerce")

    data_version = {
        "weather": _file_fingerprint(weather_csv),
        "merged_station_hour": _file_fingerprint(merged_csv),
        "dockless_trip": _file_fingerprint(dockless_csv),
        "row_counts": {
            "weather": int(len(weather)),
            "merged_station_hour": int(len(merged)),
            "dockless_trip": int(len(dockless)),
        },
    }

    return weather, merged, dockless, data_version



def save_data_version(data_version: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_version.json").write_text(
        json.dumps(data_version, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
