# 📚 Threatora Python API Reference

Threatora can be used both as a CLI application and as a Python library.

---

## 1. `models.autoencoder`

### `Autoencoder(input_dim, bottleneck_dim=24)`
Neural autoencoder that compresses input flow features into a bottleneck latent manifold.

```python
from models.autoencoder import Autoencoder

model = Autoencoder(input_dim=40, bottleneck_dim=24)
reconstructed = model(flow_tensors)
```

### `ReconstructionEnergyWrapper(autoencoder_model)`
Computes reconstruction error energy $E(x) = \sum (x - \hat{x})^2$ for SHAP gradient attribution.

```python
from models.autoencoder import ReconstructionEnergyWrapper
import shap

energy_model = ReconstructionEnergyWrapper(autoencoder)
explainer = shap.GradientExplainer(energy_model, background_data)
shap_values = explainer.shap_values(test_data)
```

---

## 2. `models.capsnet`

### `GTNLayer(in_features, out_features)`
Graph transformation network layer mapping spatial flow feature interactions.

### `SpatialTemporalCapsNet(input_dim, hidden_dim=32, num_capsules=8, cap_dim=8, num_classes=2)`
End-to-end spatial-temporal representation learner.

```python
from models.capsnet import SpatialTemporalCapsNet

capsnet = SpatialTemporalCapsNet(input_dim=20, num_capsules=8, cap_dim=8)
# Extract 64-dimensional dynamic routing vector
embeddings = capsnet.extract_features(flow_tensors)
```

---

## 3. `models.classifier`

### `train_xgboost(train_features, y_train, n_estimators=200, max_depth=6, lr=0.07, threads=6)`
Trains multi-threaded histogram-based XGBoost classifier on capsule representations.

### `evaluate_predictions(model, test_features, y_test, class_names=None)`
Computes accuracy, precision, recall, F1, ROC-AUC, MCC, false alarm rate, and confusion matrix.
