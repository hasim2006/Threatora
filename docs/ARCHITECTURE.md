# 🏛️ Threatora Deep Architecture & Theory

Threatora solves the fundamental challenges of modern Network Intrusion Detection Systems (NIDS):
1. **Extreme Dimensionality & Redundancy**: Hundreds of statistical flow features.
2. **Spatial-Temporal Non-Linear Dependencies**: Inter-packet dynamics and header spatial topology.
3. **Severe Class Imbalance**: Attack flows constitute less than 1% of enterprise network traffic.
4. **Latency Constraints**: High throughput line rates require inference under 0.1 ms per flow.

---

## 🔬 Four-Tier Hybrid Pipeline

```
Raw Network Flows ──> [ Tier 1: Deep Autoencoder + SHAP ]
                             │
                             ▼ (Top-20 Attributed Features)
                      [ Tier 2: Spatial-Temporal CapsNet ]
                             │
                             ▼ (64D Squash Capsule Embeddings)
                      [ Tier 3: Multi-Core Histogram XGBoost ]
                             │
                             ▼
                      [ Tier 4: Zero-Trust Decision Engine & Diagnostics ]
```

---

### Tier 1: Autoencoder Reconstruction Energy & SHAP Attribution

Network intrusions deviate geometrically from the manifold of benign network behaviors $\mathcal{M}$. The Deep Autoencoder models normal flow reconstruction:

$$\hat{x} = \text{Dec}(\text{Enc}(x))$$

The total reconstruction energy scalar:
$$E(x) = \sum_{i=1}^{D} \left( x_i - \hat{x}_i \right)^2$$

Using **SHAP GradientExplainer**, the exact attribution of each feature $i$ is calculated via gradients with respect to the input:

$$\phi_i(x) = (x_i - r_i) \int_{0}^{1} \frac{\partial E(r + \alpha(x - r))}{\partial x_i} d\alpha$$

The top-$k$ features ($k=20$) with maximum mean absolute attribution $\mathbb{E}[|\phi_i|]$ are selected, discarding noisy, non-informative telemetry.

---

### Tier 2: Spatial-Temporal Graph Transformation & Capsule Squash Routing

1. **Spatial Representation (GTN Layer)**:
   $$Z_{\text{spatial}} = \text{ReLU}(X_{\text{top\_k}} W_s + b_s)$$
   Learns spatial correlations between disparate packet flow attributes (e.g., packet lengths, window sizes, TTL).

2. **Temporal Modeling (LSTM Layer)**:
   $$h_t = \text{LSTM}(Z_{\text{spatial}, t}, h_{t-1})$$
   Captures sequential packet burst kinetics and temporal traffic spikes.

3. **Capsule Vector Squashing**:
   Vectors encapsulate entity properties (existence probability and spatial orientation). The non-linear squash function normalizes length to $[0, 1)$:

   $$v_j = \frac{\|s_j\|^2}{1 + \|s_j\|^2} \frac{s_j}{\|s_j\|}$$

   where $s_j \in \mathbb{R}^8$ across 8 primary capsules, yielding a dense, highly clustered **64-dimensional dynamic routing vector**.

---

### Tier 3: Multi-Core Histogram XGBoost

The 64-dimensional capsule representations are fed into a parallelized histogram-based Extreme Gradient Boosting classifier:

$$\mathcal{L}^{(t)} = \sum_{i=1}^n \left[ g_i f_t(v_i) + \frac{1}{2} h_i f_t^2(v_i) \right] + \Omega(f_t)$$

- **Tree Method**: `hist` (Histogram-optimized for memory efficiency and multi-core CPU scaling).
- **Subsampling & Column Sampling**: $0.85$ to prevent overfitting on specific attack variants.
- **Inference Latency**: Sub-millisecond evaluation suitable for software-defined network (SDN) gateways.

---

### Tier 4: Diagnostic Metrics & Calibration

Every evaluation step produces calibration curves (Brier score), Detection Error Tradeoff (DET), and Matthews Correlation Coefficient (MCC) to evaluate performance under high class imbalance.
