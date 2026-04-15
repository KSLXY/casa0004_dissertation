# Conclusion Draft

This enhancement round integrates re-clustering, neighbor flow spillover, and a full event calendar.
For weekday demand, the best full-scenario model is `RandomForest` (R2=0.456, MAE=1.169, RMSE=1.922).
For weekend demand, the best full-scenario model is `RandomForest` (R2=0.272, MAE=0.879, RMSE=1.372).
Ablation outputs quantify the incremental contribution of cluster, neighbor, and event features.

## Improvement vs run_20260415_102043
- `weekday` R2 change vs baseline: +0.1016
- `weekend` R2 change vs baseline: +0.0187