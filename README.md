<div align="center">

# 🛡️ Threatora: Advanced Cyber Attack & Intrusion Detection System

<p align="center">
  <strong>An End-to-End Multi-Tier Hybrid Deep Learning & Gradient Boosting Framework for Line-Rate Network Threat Detection</strong>
</p>

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Parallel%20Hist-1B72BA?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](CONTRIBUTING.md)

</div>

---

## 📖 Table of Contents
- [Executive Overview](#-executive-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Theoretical & Mathematical Formulations](#-theoretical--mathematical-formulations)
- [Empirical Benchmark Results](#-empirical-benchmark-results)
- [Telemetry & Diagnostic Visualizations](#-telemetry--diagnostic-visualizations)
- [Repository Layout](#-repository-layout)
- [Quick Start & Installation](#-quick-start--installation)
- [CLI Reference](#-cli-reference)
- [Docker Deployment](#-docker-deployment)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Citation](#-citation)
- [License & Authors](#-license--authors)

---

## 🎯 Executive Overview

Modern enterprise network environments face unprecedented attack velocities, ranging from automated botnets and distributed denial-of-service (DDoS) campaigns to zero-day exploits and subtle multi-stage infiltrations. Traditional rule-based Intrusion Detection Systems (IDS) suffer from high false-alarm rates and are unable to detect unknown metamorphic threats.

**Threatora** is a state-of-the-art intrusion detection research framework designed to deliver **ultra-low false alarm rates**, **sub-millisecond line-rate latency**, and **robust detection accuracy**. Evaluated rigorously against millions of packet flows from the **UNSW-NB15** and **CSE-CIC-IDS2018** benchmarks, Threatora combines:
1. **Unsupervised Deep Reconstruction & SHAP Gradient Attribution** for noise reduction and dynamic signature selection.
2. **Spatial-Temporal Graph Transformation & Capsule Squashing Networks** for learning packet trajectory manifolds without positional loss.
3. **Multi-Threaded Histogram XGBoost** for deterministic, low-latency zero-trust decision making.

---

## ⚡ Key Features

- 🧬 **Reconstruction Energy Attribution**: Discards irrelevant protocol noise by selecting the top-20 attack-discriminative features using exact SHAP gradients on Autoencoder reconstruction energy.
- 🌐 **Spatial-Temporal Capsule Embedding**: Combines Graph Transformation Networks (GTN) and Bidirectional/Unidirectional LSTMs with vector squash non-linearities, preserving complex flow orientations.
- 🚀 **Hardware-Accelerated CPU Parallelism**: Leverages multi-core histogram-based gradient boosting (`tree_method='hist'`), enabling deployment on high-throughput software-defined networking (SDN) appliances without dedicated enterprise GPUs.
- 🔬 **Publication-Grade Diagnostic Telemetry**: Automatically renders 300-DPI publication-ready ROC curves, PR curves, Detection Error Tradeoff (DET) plots, and probability calibration manifolds.
- 🐳 **Containerized & Reproducible**: Fully Dockerized with Docker Compose support and reproducible random seeds.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["Tier 0: Flow Ingestion & Sanitization"]
        A[Raw Network PCAP / CSV Flows] --> B[Sanitization: Inf/NaN Cleaning]
        B --> C[Stratified 75/25 Train-Test Split]
        C --> D[Standard Scaling Normalization]
    end

    subgraph Tier1["Tier 1: Feature Extraction & Gradient Attribution"]
        D --> E[Deep Autoencoder Reconstruction]
        E --> F["Reconstruction Energy: E(x) = Σ (x - x̂)²"]
        F --> G[SHAP GradientExplainer Attribution]
        G --> H[Top-20 Salient Feature Projection]
    end

    subgraph Tier2["Tier 2: Spatial-Temporal Capsule Network"]
        H --> I[GTN Spatial Transformation Layer]
        I --> J[LSTM Temporal Packet Sequence Modeling]
        J --> K[Dynamic Capsule Projection: 8 Capsules x 8D]
        K --> L["Squash Activation: v = (||s||² / (1 + ||s||²)) * (s / ||s||)"]
        L --> M[64-Dimensional Latent Capsule Vector]
    end

    subgraph Tier3["Tier 3: Multi-Threaded Classification"]
        M --> N[Histogram-based Multi-Core XGBoost]
        N --> O[Zero-Trust Threat Probability Score]
    end

    subgraph Tier4["Tier 4: Telemetry & Decision"]
        O --> P{Threat Threshold >= 0.50}
        P -->|Benign| Q[Forward Traffic / Audit Log]
        P -->|Attack| R[Drop Packet & Alert SOC / SIEM]
        O --> S[Generate 300-DPI Publication Diagnostics]
    end
```

---

## 🔬 Theoretical & Mathematical Formulations

### 1. Autoencoder Energy-based SHAP Selection
Let normal network behavior lie on a low-dimensional manifold $\mathcal{M} \subset \mathbb{R}^D$. An autoencoder parameters $\theta = \{W_e, b_e, W_d, b_d\}$ are optimized to minimize reconstruction error:

$$\min_{\theta} \mathbb{E}_{x \sim \mathcal{M}} \left[ \| x - \text{Dec}_{\theta}(\text{Enc}_{\theta}(x)) \|_2^2 \right]$$

The reconstruction energy scalar represents the distance to the normal traffic manifold:

$$E(x) = \sum_{i=1}^D (x_i - \hat{x}_i)^2$$

SHAP feature importance gradients $\nabla_x E(x)$ are computed through backpropagation across the entire feature space to identify the top $k=20$ attack indicators:

$$\phi_i(x) = (x_i - r_i) \int_0^1 \frac{\partial E(r + \alpha(x - r))}{\partial x_i} d\alpha$$

### 2. Capsule Squash Activation
Unlike standard scalar neurons with ReLU/Sigmoid activation, capsule networks encode properties into vector orientations. The non-linear squash function constrains capsule lengths to $[0, 1)$:

$$\mathbf{v}_j = \frac{\|\mathbf{s}_j\|^2}{1 + \|\mathbf{s}_j\|^2} \frac{\mathbf{s}_j}{\|\mathbf{s}_j\| + \epsilon}$$

where $\mathbf{s}_j \in \mathbb{R}^8$ denotes total input to capsule $j$. The resultant 64-dimensional feature vector captures intricate attack trajectories invariant to flow noise.

---

## 📊 Empirical Benchmark Results

Threatora was thoroughly benchmarked on two industry-standard datasets under rigorous out-of-sample test conditions:

| Metric | UNSW-NB15 Dataset | CSE-CIC-IDS2018 (Botnet) | Description |
| :--- | :---: | :---: | :--- |
| **Total Evaluated Flows** | 257,673 flows | 1,048,575 flows | Full dataset volume processed |
| **Out-of-Sample Test Records** | 175,341 flows | 62,500 flows | Unseen test partition |
| **Overall Accuracy (ACC)** | **90.41%** | **99.99%** | Correctly classified traffic ratio |
| **Attack Detection Rate (Recall)** | **98.24%** | **99.97%** | True positive intrusion detection |
| **Precision (PPV)** | **90.04%** | **99.99%** | Confidence in positive attack alarms |
| **F1-Score** | **0.9396** | **0.9998** | Harmonic mean of precision & recall |
| **ROC-AUC Score** | **0.9723** | **1.0000** | Area under the ROC curve |
| **False Positive Rate (FPR)** | **8.12%** | **0.004%** | Only **2 false alarms** in 50,000 flows |
| **Missed Attack Rate (FNR)** | **1.76%** | **0.029%** | Only **5 missed attacks** in 17,000 flows |
| **Inference Latency** | < 0.09 ms / flow | < 0.08 ms / flow | Multi-core CPU histogram acceleration |

---

## 📈 Telemetry & Diagnostic Visualizations

Each pipeline run automatically exports high-resolution (300 DPI) publication-grade diagnostic suites into `./outputs/`:

### CSE-CIC-IDS2018 Botnet Detection Telemetry
| Benchmark Diagnostics | Probability Margin & Calibration | Neural Learning Curves |
| :---: | :---: | :---: |
| <img src="outputs/cicids2018/cicids2018_benchmark_results.png" width="350"/> | <img src="outputs/cicids2018/cicids2018_diagnostic_dashboard.png" width="350"/> | <img src="outputs/cicids2018/cicids2018_learning_curves.png" width="350"/> |

### UNSW-NB15 Network Anomaly Telemetry
| Benchmark Diagnostics | Probability Margin & Calibration | Neural Learning Curves |
| :---: | :---: | :---: |
| <img src="outputs/unsw_nb15/unsw_nb15_benchmark_results.png" width="350"/> | <img src="outputs/unsw_nb15/unsw_nb15_diagnostic_dashboard.png" width="350"/> | <img src="outputs/unsw_nb15/unsw_nb15_learning_curves.png" width="350"/> |

---

## 📁 Repository Layout

```
Threatora/
├── .github/
│   ├── workflows/
│   │   └── ci.yml                 # Automated testing & linting pipeline
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md          # Structured bug reports
│   │   └── feature_request.md     # Feature and model enhancement requests
│   └── pull_request_template.md   # PR review checklist
├── configs/
│   └── default_config.yaml        # System, dataset, and model hyperparameters
├── docs/
│   ├── ARCHITECTURE.md            # Deep theoretical and architectural specifications
│   ├── BENCHMARKS.md              # Granular benchmark breakdown and comparative analysis
│   └── API_REFERENCE.md           # Python library interface documentation
├── models/
│   ├── __init__.py
│   ├── autoencoder.py             # Deep Autoencoder & SHAP energy wrapper
│   ├── capsnet.py                 # GTN layer, LSTM sequence, & Capsule squash network
│   └── classifier.py             # Multi-threaded histogram XGBoost & metric engine
├── utils/
│   ├── __init__.py
│   ├── data_loader.py             # Preprocessing, KaggleHub auto-download, & data cleaning
│   └── visualizer.py              # 300-DPI publication dashboard generator
├── outputs/                       # Rendered benchmark telemetry figures
│   ├── cicids2018/
│   └── unsw_nb15/
├── tests/                         # Pytest unit tests
│   ├── test_autoencoder.py
│   ├── test_capsnet.py
├── pipelines/          # Dataset Execution Pipelines
│   ├── __init__.py
│   ├── pipeline_cicids2018.py     # Standalone CSE-CIC-IDS2018 pipeline
│   └── pipeline_unsw_nb15.py      # Standalone UNSW-NB15 pipeline
├── tests/                         # Unit test suite
│   ├── __init__.py
│   ├── test_autoencoder.py
│   ├── test_capsnet.py
│   ├── test_classifier.py
│   ├── test_data_loader.py
│   └── test_pipelines.py
├── threatora/                     # Top-level Python package interface
│   └── __init__.py
├── .gitattributes                 # Line ending normalization
├── .gitignore                      # Safe exclusions (no raw datasets, no cache)
├── CONTRIBUTING.md                 # Contribution guidelines
├── CODE_OF_CONDUCT.md             # Contributor Covenant standard
├── Dockerfile                     # Container deployment image
├── docker-compose.yml             # Container orchestration
├── LICENSE                        # MIT Open Source License
├── Makefile                       # Developer task automation
├── main_unified_pipeline.py       # Master CLI execution runner
├── pyproject.toml                 # Modern PEP 517 build configuration
├── requirements.txt               # Pinned dependencies
├── SECURITY.md                    # Vulnerability reporting disclosure policy
└── setup.py                       # Packaging and installation script
```

---

## 🚀 Quick Start & Installation

### Option 1: Standard Installation
```bash
# 1. Clone repository
git clone https://github.com/hasim2006/Threatora.git
cd Threatora

# 2. Create virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Option 2: Docker Installation
```bash
docker build -t threatora:latest .
docker run --rm -v $(pwd)/outputs:/app/outputs threatora:latest --dataset all
```

---

## 💻 CLI Reference

Run the unified benchmark runner across datasets:

```bash
# Evaluate both UNSW-NB15 and CSE-CIC-IDS2018 end-to-end
python main_unified_pipeline.py --dataset all

# Evaluate UNSW-NB15 only
python main_unified_pipeline.py --dataset unsw

# Evaluate CSE-CIC-IDS2018 only
python main_unified_pipeline.py --dataset cicids

# Adjust sample volume and output directory
python main_unified_pipeline.py --dataset cicids --max-samples 500000 --output-dir ./custom_outputs
```

Or execute dedicated dataset pipelines directly:
```bash
python pipelines/pipeline_unsw_nb15.py
python pipelines/pipeline_cicids2018.py
```

---

## 🧪 Testing & Quality Assurance

Threatora includes a comprehensive unit testing suite covering layer shapes, gradient backpropagation, squashing boundaries, and classifier metrics:

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage report
pytest --cov=models --cov=utils tests/
```

---

## 📜 Citation

If you use Threatora or its architectural concepts in your research, please cite:

```bibtex
@software{hasim2026threatora,
  author       = {Mohammad Hasim},
  title        = {Threatora: Advanced Cyber Attack & Intrusion Detection System},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/hasim2006/Threatora}}
}
```

---

## 👤 Author & Maintainer

**Mohammad Hasim**
- GitHub: [@hasim2006](https://github.com/hasim2006)
- Email: `juug25btech29109@jainuniversity.ac.in`

Distributed under the [MIT License](LICENSE).
