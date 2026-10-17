"""
High-Resolution (300 DPI) Publication-Grade Visualizer for Intrusion Detection
Generates:
1. Benchmark Evaluation Dashboard (Confusion Matrix, ROC, PR, SHAP Importance)
2. Cyber Attack Diagnostic Dashboard (Separation Margin, Correlation, DET, Calibration)
3. Neural Learning Convergence Curves (Autoencoder MSE Loss & CapsNet Loss)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

def generate_all_visualizations(y_test, metrics, selected_features, mean_abs_shap, selected_indices,
                                X_test_sel, ae_losses, net_losses, output_dir=".", prefix=""):
    """
    Generates all 3 comprehensive visual dashboard suites.
    """
    os.makedirs(output_dir, exist_ok=True)
    cm = metrics["cm"]
    y_prob = metrics["y_prob"]
    roc_auc = metrics["roc_auc"]

    pfx = f"{prefix}_" if prefix else ""
    out_main = os.path.join(output_dir, f"{pfx}benchmark_results.png")
    out_diag = os.path.join(output_dir, f"{pfx}diagnostic_dashboard.png")
    out_learn = os.path.join(output_dir, f"{pfx}learning_curves.png")

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # =========================================================================
    # 1. Benchmark Evaluation Dashboard
    # =========================================================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12), dpi=250)

    # (a) Normalized Confusion Matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot_matrix = [[f"{val:,}\n({pct:.1%})" for val, pct in zip(row_val, row_pct)]
                    for row_val, row_pct in zip(cm, cm_norm)]

    sns.heatmap(cm, annot=annot_matrix, fmt="", cmap="Blues", ax=axes1[0, 0],
                cbar=True, annot_kws={"size": 13, "weight": "bold"},
                xticklabels=['Benign (0)', 'Attack (1)'], yticklabels=['Benign (0)', 'Attack (1)'])
    axes1[0, 0].set_title("(a) Confusion Matrix with Normalization", fontsize=14, fontweight="bold", pad=12)
    axes1[0, 0].set_xlabel("Predicted Label", fontsize=12, labelpad=8)
    axes1[0, 0].set_ylabel("True Label", fontsize=12, labelpad=8)

    # (b) ROC Curve
    fpr_vals, tpr_vals, _ = roc_curve(y_test, y_prob)
    axes1[0, 1].plot(fpr_vals, tpr_vals, color="#1f77b4", lw=2.5, label=f"GTN-CapsNet + XGBoost (AUC = {roc_auc:.4f})")
    axes1[0, 1].plot([0, 1], [0, 1], linestyle="--", color="#888888", lw=1.5, label="Random Guess (AUC = 0.5000)")
    axes1[0, 1].fill_between(fpr_vals, tpr_vals, alpha=0.15, color="#1f77b4")
    axes1[0, 1].set_title("(b) Receiver Operating Characteristic (ROC-AUC)", fontsize=14, fontweight="bold", pad=12)
    axes1[0, 1].set_xlabel("False Positive Rate (FPR)", fontsize=12, labelpad=8)
    axes1[0, 1].set_ylabel("True Positive Rate (Recall)", fontsize=12, labelpad=8)
    axes1[0, 1].legend(loc="lower right", fontsize=11, frameon=True)
    axes1[0, 1].grid(True, alpha=0.4)

    # (c) Precision-Recall Curve
    pr_vals, rec_vals, _ = precision_recall_curve(y_test, y_prob)
    axes1[1, 0].plot(rec_vals, pr_vals, color="#2ca02c", lw=2.5, label=f"F1-Score = {metrics['f1']:.4f}")
    axes1[1, 0].fill_between(rec_vals, pr_vals, alpha=0.15, color="#2ca02c")
    axes1[1, 0].set_title("(c) Precision-Recall Curve", fontsize=14, fontweight="bold", pad=12)
    axes1[1, 0].set_xlabel("Recall (Detection Rate)", fontsize=12, labelpad=8)
    axes1[1, 0].set_ylabel("Precision (Positive Predictive Value)", fontsize=12, labelpad=8)
    axes1[1, 0].legend(loc="lower left", fontsize=11, frameon=True)
    axes1[1, 0].grid(True, alpha=0.4)

    # (d) Top SHAP Feature Importance
    feat_importances = mean_abs_shap[selected_indices]
    sorted_pos = np.argsort(feat_importances)
    y_ticks = np.arange(len(sorted_pos))
    y_labels = [selected_features[i] for i in sorted_pos]
    
    axes1[1, 1].barh(y_ticks, feat_importances[sorted_pos], color="#ff7f0e", edgecolor="#333333", alpha=0.85)
    axes1[1, 1].set_yticks(y_ticks)
    axes1[1, 1].set_yticklabels(y_labels, fontsize=10)
    axes1[1, 1].set_title("(d) Top Feature Attributions via SHAP Explainer", fontsize=14, fontweight="bold", pad=12)
    axes1[1, 1].set_xlabel("Mean |SHAP Value| (Energy Attribution)", fontsize=12, labelpad=8)
    axes1[1, 1].grid(True, alpha=0.4)

    plt.tight_layout(pad=3.0)
    fig1.savefig(out_main, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"[+] Saved Benchmark Figure: {out_main}", flush=True)

    # =========================================================================
    # 2. Cyber Attack Diagnostic Dashboard
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12), dpi=250)

    # (a) Predicted Probability Distribution
    norm_probs = y_prob[y_test == 0]
    attk_probs = y_prob[y_test == 1]
    
    sns.histplot(norm_probs, bins=40, kde=True, color="#1f77b4", label="Benign Traffic (0)", ax=axes2[0, 0], stat="density", alpha=0.55)
    sns.histplot(attk_probs, bins=40, kde=True, color="#d62728", label="Malicious Attacks (1)", ax=axes2[0, 0], stat="density", alpha=0.55)
    axes2[0, 0].axvline(0.5, color="black", linestyle="--", lw=1.8, label="Decision Threshold (0.5)")
    axes2[0, 0].set_title("(a) Attack vs. Benign Probability Separation Margin", fontsize=14, fontweight="bold", pad=12)
    axes2[0, 0].set_xlabel("Predicted Attack Probability $P(y=1)$", fontsize=12, labelpad=8)
    axes2[0, 0].set_ylabel("Probability Density", fontsize=12, labelpad=8)
    axes2[0, 0].legend(fontsize=11, frameon=True)
    axes2[0, 0].grid(True, alpha=0.4)

    # (b) Feature Correlation Heatmap of Top 10 Features
    top_10_names = selected_features[:10]
    top_10_data = pd.DataFrame(X_test_sel[:, :10], columns=top_10_names)
    corr_matrix = top_10_data.corr()

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes2[0, 1],
                cbar=True, annot_kws={"size": 10}, linewidths=0.5)
    axes2[0, 1].set_title("(b) Cross-Feature Correlation Matrix (Top 10 Features)", fontsize=14, fontweight="bold", pad=12)
    axes2[0, 1].tick_params(axis='x', rotation=45)
    axes2[0, 1].tick_params(axis='y', rotation=0)

    # (c) Detection Error Tradeoff (DET) Curve
    axes2[1, 0].plot(fpr_vals, 1.0 - tpr_vals, color="#9467bd", lw=2.5, label="DET Curve (FPR vs Miss Rate)")
    axes2[1, 0].scatter([metrics["fpr"]], [metrics["fnr"]], color="red", s=80, zorder=5, 
                        label=f"Operating Point (FPR={metrics['fpr']*100:.2f}%, FNR={metrics['fnr']*100:.2f}%)")
    axes2[1, 0].set_title("(c) Detection Error Tradeoff (DET Curve)", fontsize=14, fontweight="bold", pad=12)
    axes2[1, 0].set_xlabel("False Alarm Rate (FPR)", fontsize=12, labelpad=8)
    axes2[1, 0].set_ylabel("Miss Detection Rate (FNR)", fontsize=12, labelpad=8)
    axes2[1, 0].legend(fontsize=11, frameon=True)
    axes2[1, 0].grid(True, alpha=0.4)

    # (d) Reliability / Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    axes2[1, 1].plot(prob_pred, prob_true, marker='o', color="#e377c2", lw=2.2, label="Model Reliability")
    axes2[1, 1].plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5, label="Perfect Calibration")
    axes2[1, 1].set_title("(d) Probability Calibration & Reliability Curve", fontsize=14, fontweight="bold", pad=12)
    axes2[1, 1].set_xlabel("Mean Predicted Probability", fontsize=12, labelpad=8)
    axes2[1, 1].set_ylabel("Empirical Fraction of Attacks", fontsize=12, labelpad=8)
    axes2[1, 1].legend(fontsize=11, frameon=True)
    axes2[1, 1].grid(True, alpha=0.4)

    plt.tight_layout(pad=3.0)
    fig2.savefig(out_diag, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"[+] Saved Diagnostic Dashboard: {out_diag}", flush=True)

    # =========================================================================
    # 3. Neural Learning Convergence Curves
    # =========================================================================
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 6), dpi=250)

    # (a) Autoencoder Loss Curve
    ae_epochs = range(1, len(ae_losses) + 1)
    axes3[0].plot(ae_epochs, ae_losses, marker='s', color="#1f77b4", lw=2.5, label="Autoencoder MSE Loss")
    axes3[0].set_title("(a) Autoencoder Reconstruction Loss Convergence", fontsize=13, fontweight="bold", pad=10)
    axes3[0].set_xlabel("Epoch", fontsize=11)
    axes3[0].set_ylabel("Mean Squared Error (MSE)", fontsize=11)
    axes3[0].legend(fontsize=11, frameon=True)
    axes3[0].grid(True, alpha=0.4)

    # (b) Spatial-Temporal CapsNet Loss Curve
    net_epochs = range(1, len(net_losses) + 1)
    axes3[1].plot(net_epochs, net_losses, marker='o', color="#2ca02c", lw=2.5, label="CapsNet Cross-Entropy Loss")
    axes3[1].set_title("(b) Spatial-Temporal Representation Learning Loss", fontsize=13, fontweight="bold", pad=10)
    axes3[1].set_xlabel("Epoch", fontsize=11)
    axes3[1].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes3[1].legend(fontsize=11, frameon=True)
    axes3[1].grid(True, alpha=0.4)

    plt.tight_layout(pad=2.5)
    fig3.savefig(out_learn, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"[+] Saved Learning Curves: {out_learn}", flush=True)

    return out_main, out_diag, out_learn
