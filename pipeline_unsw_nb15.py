"""
=============================================================================
   UNSW-NB15 CYBER ATTACK DETECTION PIPELINE
   Architecture: Autoencoder + SHAP -> GTN-LSTM-CapsNet -> Multi-threaded XGBoost
=============================================================================
"""

import os
import gc
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shap

from models.autoencoder import Autoencoder, ReconstructionEnergyWrapper
from models.capsnet import SpatialTemporalCapsNet
from models.classifier import train_xgboost, evaluate_predictions
from utils.data_loader import load_unsw_nb15
from utils.visualizer import generate_all_visualizations

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def configure_hardware():
    cpu_count = os.cpu_count() or 4
    threads = max(1, min(6, cpu_count))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, threads

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def perform_feature_selection(X_train, X_test, feature_names, device, top_k=20):
    print("\n" + "=" * 70, flush=True)
    print(">> STEP 1: TRAINING AUTOENCODER & SHAP GRADIENT EXPLAINER", flush=True)
    print("=" * 70, flush=True)
    
    input_dim = X_train.shape[1]
    autoencoder = Autoencoder(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.005, weight_decay=1e-5)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(X_train))
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    epochs = 8
    ae_losses = []
    autoencoder.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for bx, _ in loader:
            bx = bx.to(device)
            optimizer.zero_grad()
            recon = autoencoder(bx)
            loss = criterion(recon, bx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
        
        ep_loss = total_loss / len(dataset)
        ae_losses.append(ep_loss)
        print(f"   Epoch [{epoch}/{epochs}] - Autoencoder MSE Reconstruction Loss: {ep_loss:.5f}", flush=True)

    print("\n[*] Computing SHAP Gradient Explainer Attributions...", flush=True)
    sample_size = min(200, X_train.shape[0])
    sample_idx = np.random.choice(X_train.shape[0], size=sample_size, replace=False)
    X_sample_tensor = torch.from_numpy(X_train[sample_idx]).to(device)

    energy_model = ReconstructionEnergyWrapper(autoencoder).to(device)
    energy_model.eval()

    explainer = shap.GradientExplainer(energy_model, X_sample_tensor)
    shap_raw = explainer.shap_values(X_sample_tensor)
    if isinstance(shap_raw, list):
        shap_raw = shap_raw[0]

    shap_raw = np.asarray(shap_raw)
    if shap_raw.ndim > 2:
        shap_raw = shap_raw.squeeze()

    mean_abs_shap = np.abs(shap_raw).mean(axis=0).flatten()
    sorted_idx = np.argsort(mean_abs_shap)[-top_k:][::-1]
    selected_indices = [int(i) for i in sorted_idx]
    selected_features = [feature_names[i] for i in selected_indices]

    print(f"[+] Top {top_k} Selected Features for UNSW-NB15:", flush=True)
    for r, (idx, name) in enumerate(zip(selected_indices, selected_features), start=1):
        print(f"   {r:2d}. {name:<20} (Mean |SHAP| = {mean_abs_shap[idx]:.5f})", flush=True)

    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices]

    del autoencoder, energy_model, loader, dataset
    gc.collect()
    return X_train_sel, X_test_sel, selected_features, mean_abs_shap, selected_indices, ae_losses

def train_representation_model(X_train, y_train, X_test, device, epochs=10):
    print("\n" + "=" * 70, flush=True)
    print(">> STEP 2: TRAINING SPATIAL-TEMPORAL CAPSULE NETWORK", flush=True)
    print("=" * 70, flush=True)
    
    input_dim = X_train.shape[1]
    model = SpatialTemporalCapsNet(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    tx = torch.from_numpy(X_train)
    ty = torch.from_numpy(y_train)
    dataset = TensorDataset(tx, ty)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    net_losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bx.size(0)
            correct += (out.argmax(dim=1) == by).sum().item()
            total += by.size(0)
        
        ep_loss = total_loss / total
        ep_acc = correct / total * 100
        net_losses.append(ep_loss)
        print(f"   Epoch [{epoch:2d}/{epochs}] - Loss: {ep_loss:.5f} | Training ACC: {ep_acc:.2f}%", flush=True)

    model.eval()
    with torch.no_grad():
        train_caps = model.extract_features(tx.to(device)).cpu().numpy()
        test_caps = model.extract_features(torch.from_numpy(X_test).to(device)).cpu().numpy()

    del model, loader, dataset
    gc.collect()
    return train_caps, test_caps, net_losses

def run_unsw_pipeline(output_dir="./outputs/unsw_nb15"):
    t0 = time.time()
    set_seed(42)
    device, threads = configure_hardware()

    print("=" * 70, flush=True)
    print(">> STARTING UNSW-NB15 CYBER ATTACK DETECTION PIPELINE", flush=True)
    print(f"   * Device: {device.type.upper()} | CPU Threads: {threads}", flush=True)
    print("=" * 70, flush=True)

    X_train, y_train, X_test, y_test, feature_names = load_unsw_nb15()

    X_tr_sel, X_te_sel, sel_features, mean_abs_shap, sel_indices, ae_losses = perform_feature_selection(
        X_train, X_test, feature_names, device, top_k=20
    )

    tr_caps, te_caps, net_losses = train_representation_model(
        X_tr_sel, y_train, X_te_sel, device, epochs=10
    )

    print("\n" + "=" * 70, flush=True)
    print(">> STEP 3: TRAINING MULTI-CORE XGBOOST CLASSIFIER", flush=True)
    print("=" * 70, flush=True)
    xgb_model, fit_time = train_xgboost(tr_caps, y_train, threads=threads)
    print(f"[+] XGBoost Trained in {fit_time:.2f}s", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(">> STEP 4: OUT-OF-SAMPLE TEST EVALUATION", flush=True)
    print("=" * 70, flush=True)
    metrics = evaluate_predictions(xgb_model, te_caps, y_test, class_names=["Normal (0)", "Attack (1)"])

    print(f"   * Accuracy (ACC)        : {metrics['acc']*100:.2f}%", flush=True)
    print(f"   * Precision (Attack)    : {metrics['prec']*100:.2f}%", flush=True)
    print(f"   * Recall / Detection    : {metrics['rec']*100:.2f}%", flush=True)
    print(f"   * F1-Score              : {metrics['f1']:.4f}", flush=True)
    print(f"   * ROC-AUC Score         : {metrics['roc_auc']:.4f}", flush=True)
    print(f"   * False Alarm Rate(FPR) : {metrics['fpr']*100:.2f}% ({metrics['fp']} false alarms)", flush=True)
    print(f"   * Miss Detection (FNR)  : {metrics['fnr']*100:.2f}% ({metrics['fn']} missed attacks)", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(">> STEP 5: GENERATING VISUALIZATION DASHBOARDS", flush=True)
    print("=" * 70, flush=True)
    generate_all_visualizations(
        y_test=y_test,
        metrics=metrics,
        selected_features=sel_features,
        mean_abs_shap=mean_abs_shap,
        selected_indices=sel_indices,
        X_test_sel=X_te_sel,
        ae_losses=ae_losses,
        net_losses=net_losses,
        output_dir=output_dir,
        prefix="unsw_nb15"
    )

    print(f"\n[***] UNSW-NB15 Pipeline Finished in {time.time()-t0:.2f}s [***]\n", flush=True)
    return metrics

if __name__ == "__main__":
    run_unsw_pipeline()
