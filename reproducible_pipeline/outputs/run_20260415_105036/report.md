# Casa0004 Reproducible Pipeline Report

## 1. Data Version

```json
{
  "weather": {
    "path": "july_to_dec_2024_cleaned.csv",
    "size_bytes": 327177,
    "modified_time": 1753378077.0827131,
    "sha256_head_1mb": "0c3c28cc750573f042bbf74d02c6fb4b4bdab12960bb073de61c6a327b95ad36"
  },
  "merged_station_hour": {
    "path": "30th_may/merged_docked_dockless_weather.csv",
    "size_bytes": 58694203,
    "modified_time": 1755435835.796639,
    "sha256_head_1mb": "b105581a9a5145e19d7aa0ee1aaec8c551da40767fdd60cd89ade9a34c2dbf72"
  },
  "dockless_trip": {
    "path": "30th_may/enhanced_full_dataset_20250727_0524.csv",
    "size_bytes": 44511560,
    "modified_time": 1753593854.7820597,
    "sha256_head_1mb": "10c7e4a8b4932bed6acf68c3f40851515af55ee61a07409a6b8cff55dcd571dc"
  },
  "row_counts": {
    "weather": 4416,
    "merged_station_hour": 502602,
    "dockless_trip": 103038
  }
}
```

## 2. Metrics Summary (Full Scenario)

- `weekday` best model: `RandomForest` | R2=0.4561, MAE=1.1691, RMSE=1.9215, WAPE=73.77%
- `weekend` best model: `RandomForest` | R2=0.2723, MAE=0.8788, RMSE=1.3722, WAPE=88.24%

## 3. Full Configuration Snapshot

```toml
[data]
weather_csv = "july_to_dec_2024_cleaned.csv"
merged_station_hour_csv = "30th_may/merged_docked_dockless_weather.csv"
dockless_trip_csv = "30th_may/enhanced_full_dataset_20250727_0524.csv"

[split]
train_end = "2024-11-01 00:00:00"
val_end = "2024-12-01 00:00:00"

[feature]
target_col = "total_count"
station_id_col = "station_id"
time_col = "hourly"
add_lag_features = true
lag_hours = [1, 24, 168]
add_cluster_proxy = false
use_event_features = true
events_calendar_csv = "reproducible_pipeline/events/calendar_ireland_2024h2.csv"

[model]
random_state = 42
random_forest_n_estimators = 300
random_forest_max_depth = 16
random_forest_min_samples_leaf = 3

[cluster]
algorithm = "kmeans"
random_state = 42
k_min = 6
k_max = 16

[neighbor]
k_neighbors = 5
distance_metric = "euclidean"

[evaluation]
baseline_metrics_csv = "reproducible_pipeline/outputs/run_20260415_102043/metrics_all.csv"

[output]
output_dir = "reproducible_pipeline/outputs"
run_name = "auto"

```