from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .evaluate import compute_metrics


@dataclass
class TrainResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    feature_importance: pd.DataFrame



def _feature_sets(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    selected_features: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    # Avoid target leakage from direct components of total demand.
    leakage_cols = {
        "rental_count",
        "dockless_count",
        "dockless_offnet",
    }
    drop_cols = {target_col, time_col, "date", *leakage_cols}
    if selected_features is None:
        features = [c for c in df.columns if c not in drop_cols]
    else:
        features = [c for c in selected_features if c in df.columns and c not in drop_cols]
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    return num_cols, cat_cols



def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    time_col: str,
    random_state: int,
    rf_n_estimators: int,
    rf_max_depth: int,
    rf_min_samples_leaf: int,
    segment_name: str,
    scenario_name: str,
    selected_features: list[str] | None = None,
) -> TrainResult:
    num_cols, cat_cols = _feature_sets(train_df, target_col, time_col, selected_features=selected_features)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )

    models = {
        "BaselineMean": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "RandomForest": RandomForestRegressor(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    y_train = train_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()

    metric_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    fi_rows: list[dict] = []

    for model_name, estimator in models.items():
        pipe = Pipeline([("pre", preprocessor), ("model", estimator)])
        pipe.fit(train_df, y_train)
        pred = pipe.predict(test_df)

        m = compute_metrics(y_test, pred)
        metric_rows.append(
                {
                    "segment": segment_name,
                    "scenario": scenario_name,
                    "model": model_name,
                    **m,
                }
        )

        pred_frames.append(
            pd.DataFrame(
                {
                    "segment": segment_name,
                    "scenario": scenario_name,
                    "model": model_name,
                    "datetime": test_df[time_col].values,
                    "station_id": test_df.get("station_id", np.nan).values,
                    "y_true": y_test,
                    "y_pred": pred,
                    "residual": y_test - pred,
                    "abs_residual": np.abs(y_test - pred),
                }
            )
        )

        if model_name == "RandomForest":
            importances = pipe.named_steps["model"].feature_importances_
            pre = pipe.named_steps["pre"]
            try:
                names = list(pre.get_feature_names_out())
            except Exception:
                names = [f"f_{i}" for i in range(len(importances))]
            for n, v in zip(names, importances):
                fi_rows.append(
                    {
                        "segment": segment_name,
                        "scenario": scenario_name,
                        "model": model_name,
                        "feature": n,
                        "importance": float(v),
                    }
                )

    metrics = pd.DataFrame(metric_rows).sort_values(["scenario", "segment", "R2"], ascending=[True, True, False])
    predictions = pd.concat(pred_frames, ignore_index=True)
    feature_importance = pd.DataFrame(fi_rows).sort_values(
        ["segment", "importance"], ascending=[True, False]
    )

    return TrainResult(metrics=metrics, predictions=predictions, feature_importance=feature_importance)
