"""
High-Performance Gradient Boosting Classifier and Rigorous Evaluation Metrics
"""

import time
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef, classification_report
)

def train_xgboost(train_features, y_train, n_estimators=200, max_depth=6, lr=0.07, threads=6):
    """
    Trains histogram-based XGBoost model with multi-core CPU parallelism.
    """
    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method='hist',
        n_jobs=threads,
        random_state=42,
        eval_metric='logloss'
    )
    xgb.fit(train_features, y_train)
    duration = time.time() - t0
    return xgb, duration

def evaluate_predictions(model, test_features, y_test, class_names=None):
    """
    Computes scientific metrics and returns structured evaluation dictionary.
    """
    if class_names is None:
        class_names = ["Benign (0)", "Malicious / Attack (1)"]

    y_pred = model.predict(test_features)
    y_prob = model.predict_proba(test_features)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)

    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "roc_auc": roc_auc, "mcc": mcc, "fpr": fpr, "fnr": fnr,
        "cm": cm, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "y_pred": y_pred, "y_prob": y_prob, "report": report
    }
