# Casa0004 Reproducible Pipeline

This folder turns the notebook-first project into a reproducible experiment pipeline with ablation support.

## What it does

1. Data ingest and schema validation
2. Re-cluster stations and persist cluster artifacts
3. Build station KNN neighbors and neighbor spillover features
4. Merge full event calendar features
5. Train/evaluate models for weekday and weekend separately under 5 ablation scenarios
6. Generate standard outputs:
   - `metrics_all.csv`
   - `ablation_metrics.csv`
   - `feature_importance.csv`
   - `spatial_feature_importance.csv`
   - `predictions.csv`
   - `error_analysis.csv`
   - `model_comparison.png`
   - `residual_distribution.png`
   - `conclusion_draft.md`
7. Persist experiment record:
   - `config_snapshot.toml`
   - `data_version.json`
   - `run_summary.json`
   - `report.md`
   - `cluster_artifacts/cluster_model.pkl`
   - `cluster_artifacts/station_cluster_map.csv`
   - `cluster_artifacts/cluster_summary.csv`
   - `station_neighbors.csv`

## Run

```bash
MPLCONFIGDIR=reproducible_pipeline/.mplconfig python3 reproducible_pipeline/run_pipeline.py --config reproducible_pipeline/config/default.toml
```

## Streamlit Dashboard

```bash
streamlit run reproducible_pipeline/streamlit_app.py
```

Dashboard covers:
- Pipeline process and artifacts
- Data/result tables
- Ablation comparison
- Baseline-vs-current R2 comparison
- Feature importance (including spatial/event features)
- Text explanations and conclusion draft

## Reproducibility Check

```bash
python3 reproducible_pipeline/compare_runs.py reproducible_pipeline/outputs/run_YYYYMMDD_HHMMSS reproducible_pipeline/outputs/run_YYYYMMDD_HHMMSS
```

## Notes

- Target task defaults to `station-hour total_count` prediction.
- Baseline comparison defaults to `run_20260415_102043` via config.
- Data window defaults to `2024-07` through `2024-12`.
