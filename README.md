# MLOps Lab

AI infrastructure portfolio project — vLLM on GKE with KEDA autoscaling, MLflow experiment tracking, and local AMD GPU inference.

## Architecture
```
Local Ubuntu Lab                    Google Cloud (GKE)
─────────────────                   ──────────────────────────────────
k3s (v1.34.5)                       vllm-portfolio cluster
├── MLflow 3.10.1                   ├── default-pool (e2-standard-2)
│   └── MinIO artifact store        │   └── Prometheus + Grafana
└── Ollama (AMD RX 9060 XT)         └── gpu-pool (n1-standard-4 + T4)
    └── ROCm 6.4.1                      ├── vLLM (Phi-3-mini)
                                        └── KEDA autoscaler
```

## Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| vLLM | latest | LLM inference server, OpenAI-compatible API |
| KEDA | latest | Event-driven autoscaling on request queue depth |
| Prometheus | kube-prometheus-stack | Metrics collection |
| Grafana | bundled | Dashboards |
| MLflow | 3.10.1 | Experiment tracking |
| MinIO | latest | S3-compatible artifact store |
| k3s | v1.34.5 | Local Kubernetes |
| ROCm | 6.4.1 | AMD GPU compute (RX 9060 XT, gfx1200/RDNA4) |

## Projects

### 1. vLLM on GKE with KEDA Autoscaling

Deploys vLLM as a Kubernetes Deployment on GKE with a T4 GPU node pool. KEDA scales replicas based on `vllm:num_requests_waiting` — the number of queued inference requests — rather than CPU or memory. This mirrors production GPU inference scaling at companies like CoreWeave.

**Key features:**
- GPU taint/toleration pattern — prevents non-GPU workloads from consuming the GPU node
- NVIDIA device plugin for Kubernetes GPU scheduling
- KEDA ScaledObject targeting Prometheus metrics
- OpenAI-compatible API endpoint

**KEDA scale event observed:**
```
# Under load (50 concurrent requests):
vllm-server-764685c678-4tkqn   1/1   Running   (original)
vllm-server-6b9cfbc987-kwrrs   0/1   Pending   (KEDA scale-out triggered)

ScaledObject: READY=True ACTIVE=True
```

**Manifests:** `k8s/vllm/`

**Cost:** ~$0.21/hr (e2-standard-2 control plane + n1-standard-4 spot T4)

### 2. MLflow + MinIO on k3s

MLflow experiment tracking server with MinIO as S3-compatible artifact store, deployed on local k3s.

**Diabetes regression experiment (9 runs):**

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

**Key finding:** GBM with slow learning rate (0.05) generalized best. Aggressive regularization (Ridge alpha=10) degraded R2 from 0.461 to 0.161.

**Manifests:** `k8s/mlflow/`, `k8s/minio/`

### 3. Local AMD GPU Inference

Ollama running on AMD RX 9060 XT via ROCm 6.4.1 (RDNA4/gfx1200).

- phi3:mini: ~4.3GB VRAM (27% of 16GB), GPU% 100% during inference
- Clock: 379MHz idle → 3480MHz under load
- Demonstrates ROCm as viable alternative to CUDA for LLM inference

## Repository Structure
```
mlops-lab/
├── k8s/
│   ├── vllm/
│   │   ├── vllm-deployment.yaml      # vLLM Deployment + Service
│   │   ├── vllm-podmonitor.yaml      # Prometheus PodMonitor
│   │   └── vllm-scaledobject.yaml    # KEDA ScaledObject
│   ├── mlflow/
│   │   └── mlflow.yaml               # MLflow server Deployment
│   └── minio/
│       └── minio.yaml                # MinIO Deployment + PVC
└── experiments/
    ├── diabetes_regression.py        # 9-run MLflow experiment
    └── load_test.py                  # Async load test for KEDA trigger

## Key Learnings

- MLflow 3.x requires 2Gi memory on k3s (vs 512Mi for 2.x)
- GKE 1.35 GPU nodes auto-taint with nvidia.com/gpu=present:NoSchedule
- KEDA ScaledObject pausing requires deleting pending pods from previous scale-out
- ROCm 6.4.1 works cleanly with Ollama on RDNA4 (gfx1200) — no driver patches needed
- vLLM metrics are enabled by default — --enable-metrics flag does not exist
