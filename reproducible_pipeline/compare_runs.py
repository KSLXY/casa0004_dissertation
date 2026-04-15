from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd



def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two pipeline runs by metrics_all.csv")
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    args = parser.parse_args()

    a = pd.read_csv(args.run_a / "metrics_all.csv")
    b = pd.read_csv(args.run_b / "metrics_all.csv")
    if "scenario" not in a.columns:
        a["scenario"] = "legacy"
    if "scenario" not in b.columns:
        b["scenario"] = "legacy"
    a = a.sort_values(["scenario", "segment", "model"]).reset_index(drop=True)
    b = b.sort_values(["scenario", "segment", "model"]).reset_index(drop=True)

    z = a.merge(b, on=["scenario", "segment", "model"], suffixes=("_a", "_b"))
    for c in ["MAE", "RMSE", "R2", "WAPE", "MAPE_pos"]:
        z[f"delta_{c}"] = (z[f"{c}_b"] - z[f"{c}_a"]).abs()

    print(z[["scenario", "segment", "model", "delta_R2", "delta_MAE", "delta_RMSE", "delta_WAPE"]].to_string(index=False))
    print("max_delta_R2", z["delta_R2"].max())


if __name__ == "__main__":
    main()
