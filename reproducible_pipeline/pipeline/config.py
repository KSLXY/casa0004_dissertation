from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass
class DataPaths:
    weather_csv: Path
    merged_station_hour_csv: Path
    dockless_trip_csv: Path


@dataclass
class SplitConfig:
    train_end: str
    val_end: str


@dataclass
class FeatureConfig:
    target_col: str
    station_id_col: str
    time_col: str
    add_lag_features: bool
    lag_hours: list[int]
    add_cluster_proxy: bool
    use_event_features: bool
    events_calendar_csv: Path


@dataclass
class ModelConfig:
    random_state: int
    random_forest_n_estimators: int
    random_forest_max_depth: int
    random_forest_min_samples_leaf: int


@dataclass
class ClusterConfig:
    random_state: int
    algorithm: str
    k_min: int
    k_max: int


@dataclass
class NeighborConfig:
    k_neighbors: int
    distance_metric: str


@dataclass
class EvaluationConfig:
    baseline_metrics_csv: Path


@dataclass
class OutputConfig:
    output_dir: Path
    run_name: str


@dataclass
class PipelineConfig:
    data: DataPaths
    split: SplitConfig
    feature: FeatureConfig
    model: ModelConfig
    cluster: ClusterConfig
    neighbor: NeighborConfig
    evaluation: EvaluationConfig
    output: OutputConfig



def load_config(config_path: Path) -> PipelineConfig:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    data = DataPaths(
        weather_csv=Path(raw["data"]["weather_csv"]),
        merged_station_hour_csv=Path(raw["data"]["merged_station_hour_csv"]),
        dockless_trip_csv=Path(raw["data"]["dockless_trip_csv"]),
    )
    split = SplitConfig(
        train_end=raw["split"]["train_end"],
        val_end=raw["split"]["val_end"],
    )
    feature = FeatureConfig(
        target_col=raw["feature"]["target_col"],
        station_id_col=raw["feature"]["station_id_col"],
        time_col=raw["feature"]["time_col"],
        add_lag_features=bool(raw["feature"]["add_lag_features"]),
        lag_hours=list(raw["feature"]["lag_hours"]),
        add_cluster_proxy=bool(raw["feature"]["add_cluster_proxy"]),
        use_event_features=bool(raw["feature"]["use_event_features"]),
        events_calendar_csv=Path(raw["feature"]["events_calendar_csv"]),
    )
    model = ModelConfig(
        random_state=int(raw["model"]["random_state"]),
        random_forest_n_estimators=int(raw["model"]["random_forest_n_estimators"]),
        random_forest_max_depth=int(raw["model"]["random_forest_max_depth"]),
        random_forest_min_samples_leaf=int(raw["model"]["random_forest_min_samples_leaf"]),
    )
    cluster = ClusterConfig(
        random_state=int(raw["cluster"]["random_state"]),
        algorithm=raw["cluster"]["algorithm"],
        k_min=int(raw["cluster"]["k_min"]),
        k_max=int(raw["cluster"]["k_max"]),
    )
    neighbor = NeighborConfig(
        k_neighbors=int(raw["neighbor"]["k_neighbors"]),
        distance_metric=raw["neighbor"]["distance_metric"],
    )
    evaluation = EvaluationConfig(
        baseline_metrics_csv=Path(raw["evaluation"]["baseline_metrics_csv"]),
    )
    output = OutputConfig(
        output_dir=Path(raw["output"]["output_dir"]),
        run_name=raw["output"]["run_name"],
    )

    return PipelineConfig(
        data=data,
        split=split,
        feature=feature,
        model=model,
        cluster=cluster,
        neighbor=neighbor,
        evaluation=evaluation,
        output=output,
    )
