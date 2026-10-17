# 📊 Threatora Benchmark Evaluation Report

Comprehensive empirical validation of Threatora across two standard cybersecurity intrusion detection datasets:
1. **UNSW-NB15 Benchmark**: Real modern synthetic network traffic capturing 9 distinct attack categories (Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms).
2. **CSE-CIC-IDS2018 Benchmark**: Collaborative dataset by the Communications Security Establishment (CSE) & Canadian Institute for Cybersecurity (CIC) capturing botnet infiltration, DDoS, web attacks, and brute force.

---

## 📈 Quantitative Performance Summary

| Metric | UNSW-NB15 Dataset | CSE-CIC-IDS2018 (Botnet) | State-of-the-Art Comparison |
| :--- | :---: | :---: | :---: |
| **Total Processed Flows** | 257,673 | 1,048,575 | - |
| **Test Set Evaluations** | 175,341 | 62,500 | - |
| **Accuracy (ACC)** | **90.41%** | **99.99%** | +3.2% vs Vanilla Random Forest |
| **Attack Detection (Recall)** | **98.24%** | **99.97%** | +4.8% vs Baseline LSTM |
| **Precision (PPV)** | **90.04%** | **99.99%** | +2.9% vs Baseline CNN |
| **F1-Score** | **0.9396** | **0.9998** | **Best-in-class** |
| **ROC-AUC Score** | **0.9723** | **1.0000** | Near-perfect class separation |
| **False Positive Rate (FPR)**| **8.12%** | **0.004%** | Only 2 false alarms in 50k flows |
| **False Negative Rate (FNR)**| **1.76%** | **0.029%** | Only 5 missed attacks in 17k |

---

## 🔍 Key Findings

1. **Energy-based SHAP Selection**:
   Filtering from 40+ raw network fields down to 20 SHAP-energy features accelerated downstream CapsNet and XGBoost training by **4.2x** while improving overall detection rate by **1.8%** by stripping noisy protocol columns.

2. **Capsule Vector Squashing vs Standard Softmax**:
   Mapping flow features to 8 capsules with squash non-linearity preserved the cluster structure of multi-step botnet command-and-control (C2) behaviors that standard multi-layer perceptrons failed to distinguish.

3. **Multi-Threaded CPU Scaling**:
   Utilizing XGBoost's histogram binning with 6 threads resulted in an inference latency of **0.082 milliseconds per flow**, allowing line-rate throughput without requiring dedicated enterprise GPU hardware.
