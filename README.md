# MLOps Lab

MLflow 3.10 + MinIO on k3s — AI infrastructure portfolio project.

## Stack
- MLflow 3.10.1 (tracking server + model registry)
- MinIO (S3-compatible artifact store)
- k3s v1.34.5 (local Kubernetes)
- Python 3.12 / scikit-learn

## Architecture
Python experiment → MLflow Tracking Server (k3s) → MinIO artifact store (k3s)

## Experiments

### diabetes-regression (9 runs)
Compared Ridge, RandomForest, and GradientBoosting on sklearn diabetes dataset.

| Run | RMSE | R2 |
|-----|------|----|
| gbm_lr_0.05 | 52.8 | 0.474 |
| ridge_alpha_0.1 | 53.4 | 0.461 |
| rf_50_trees | 53.5 | 0.460 |
| rf_100_trees | 53.7 | 0.456 |
| rf_200_deep | 54.5 | 0.440 |
| ridge_alpha_1.0 | 55.5 | 0.419 |
| gbm_lr_0.2 | 58.7 | 0.349 |
| ridge_alpha_10.0 | 66.7 | 0.161 |

**Key findings:**
- GBM with low learning rate (0.05) generalized best
- Aggressive regularization (Ridge alpha=10) degraded performance significantly
- RandomForest depth had diminishing returns beyond max_depth=3
- Simple linear model (Ridge 0.1) nearly matched ensemble methods

## Key Learnings
- MLflow 3.x requires 2Gi memory on k3s vs 512Mi for 2.x
- ROCm 6.4.1 on RDNA4 (RX 9060 XT) works cleanly with Ollama for local LLM inference
