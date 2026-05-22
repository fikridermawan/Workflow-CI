"""
modelling.py  (MLProject version)
==================================
Versi modelling yang dioptimalkan untuk dijalankan di dalam MLflow Project
dan GitHub Actions CI workflow.

Mendukung argumen CLI sehingga pipeline CI dapat meneruskan parameter
tanpa harus mengubah source code secara langsung.

Menjalankan secara manual:
    python modelling.py --data_dir credit_risk_preprocessing
"""

import os
import argparse
import logging
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Konstanta model ────────────────────────────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators"    : 200,
    "max_depth"       : 12,
    "min_samples_split": 5,
    "min_samples_leaf" : 2,
    "class_weight"    : "balanced",
    "random_state"    : 42,
    "n_jobs"          : -1,
}


def load_data(data_dir: str) -> tuple:
    """Memuat dataset hasil preprocessing dari folder yang ditentukan."""
    files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    for f in files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File tidak ditemukan: {path}")

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()

    log.info("Data berhasil dimuat → Train: %d | Test: %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, path: str) -> None:
    """Menyimpan confusion matrix sebagai PNG."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Non-Default", "Default"])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    ax.set_title("Confusion Matrix — Random Forest", fontweight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, path: str) -> None:
    """Menyimpan kurva ROC sebagai PNG."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc_val = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"AUC = {roc_auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Random Forest", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(data_dir: str) -> None:
    """Pipeline pelatihan dan logging utama."""
    X_train, X_test, y_train, y_test = load_data(data_dir)

    TMP = "artifacts_output"
    os.makedirs(TMP, exist_ok=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(run_name="RF_CI_Pipeline"):
        model = RandomForestClassifier(**MODEL_PARAMS)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Hitung metrik
        metrics = {
            "accuracy"    : accuracy_score(y_test, y_pred),
            "precision"   : precision_score(y_test, y_pred),
            "recall"      : recall_score(y_test, y_pred),
            "f1_score"    : f1_score(y_test, y_pred),
            "roc_auc"     : roc_auc_score(y_test, y_proba),
            "log_loss"    : log_loss(y_test, y_proba),
            "mcc"         : matthews_corrcoef(y_test, y_pred),
        }

        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        metrics["cv_auc_mean"] = cv_auc.mean()
        metrics["cv_auc_std"]  = cv_auc.std()

        # Log parameter dan metrik
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_metrics(metrics)

        for name, val in metrics.items():
            log.info("%-15s: %.4f", name, val)

        # Buat dan log artefak
        cm_path  = os.path.join(TMP, "training_confusion_matrix.png")
        roc_path = os.path.join(TMP, "roc_curve.png")
        rep_path = os.path.join(TMP, "classification_report.txt")

        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_roc_curve(y_test, y_proba, roc_path)

        report = classification_report(y_test, y_pred, target_names=["Non-Default", "Default"])
        with open(rep_path, "w") as f:
            f.write(report)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(cm_path,  artifact_path="evaluation")
        mlflow.log_artifact(roc_path, artifact_path="evaluation")
        mlflow.log_artifact(rep_path, artifact_path="evaluation")

        log.info("MLflow run selesai. Semua artefak berhasil dicatat.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training pipeline Credit Risk.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="credit_risk_preprocessing",
        help="Folder dataset hasil preprocessing.",
    )
    args = parser.parse_args()
    main(args.data_dir)
