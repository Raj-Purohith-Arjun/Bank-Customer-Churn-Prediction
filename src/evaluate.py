# src/evaluate.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve
)
from sklearn.calibration import calibration_curve


# load model
def load_model(path):
    return joblib.load(path)


# compute auc
def compute_auc(y_true, y_proba):
    return roc_auc_score(y_true, y_proba)


# compute pr auc
def compute_pr_auc(y_true, y_proba):
    return average_precision_score(y_true, y_proba)


# precision at k
def precision_at_k(y_true, y_proba, k=0.05):
    n = len(y_true)
    top_n = max(int(n * k), 1)
    idx = np.argsort(y_proba)[-top_n:]
    return np.array(y_true)[idx].mean()


# lift at k
def lift_at_k(y_true, y_proba, k=0.05):
    prec_k = precision_at_k(y_true, y_proba, k)
    base_rate = np.mean(y_true)
    return prec_k / base_rate if base_rate > 0 else 0


# plot roc
def plot_roc(y_true, y_proba, out="figs/roc.png"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}", lw=2)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# plot precision recall
def plot_pr_curve(y_true, y_proba, out="figs/pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# plot calibration
def plot_calibration(y_true, y_proba, n_bins=10, out="figs/calibration.png"):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    plt.plot(prob_pred, prob_true, "o-", label="Model")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# plot lift
def plot_lift_curve(y_true, y_proba, out="figs/lift.png"):
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(df.index, 10, labels=False, duplicates="drop")
    base_rate = df["y"].mean()
    lift = df.groupby("decile")["y"].mean() / base_rate

    plt.figure(figsize=(8, 6))
    plt.bar(range(len(lift)), lift.values, color="steelblue")
    plt.axhline(y=1, color="gray", linestyle="--")
    plt.xlabel("Decile (0=highest risk)")
    plt.ylabel("Lift")
    plt.title("Lift Chart")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# plot gain
def plot_gain_curve(y_true, y_proba, out="figs/gain.png"):
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["cum_y"] = df["y"].cumsum()
    df["cum_pct"] = df["cum_y"] / df["y"].sum()
    df["pct_pop"] = (df.index + 1) / len(df)

    plt.figure(figsize=(8, 6))
    plt.plot(df["pct_pop"], df["cum_pct"], label="Model", lw=2)
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("% Population Targeted")
    plt.ylabel("% Churners Captured")
    plt.title("Cumulative Gain Chart")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# evaluate model
def evaluate_model(model_path, data_path, figs_dir="figs"):
    os.makedirs(figs_dir, exist_ok=True)

    model = load_model(model_path)
    splits = pd.read_pickle(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    y_proba = model.predict_proba(X_test)[:, 1]

    # compute metrics
    metrics = {
        "roc_auc": compute_auc(y_test, y_proba),
        "pr_auc": compute_pr_auc(y_test, y_proba),
        "precision_at_1pct": precision_at_k(y_test, y_proba, 0.01),
        "precision_at_5pct": precision_at_k(y_test, y_proba, 0.05),
        "precision_at_10pct": precision_at_k(y_test, y_proba, 0.10),
        "lift_at_5pct": lift_at_k(y_test, y_proba, 0.05),
        "lift_at_10pct": lift_at_k(y_test, y_proba, 0.10)
    }

    # plots
    plot_roc(y_test, y_proba, f"{figs_dir}/roc.png")
    plot_pr_curve(y_test, y_proba, f"{figs_dir}/pr_curve.png")
    plot_calibration(y_test, y_proba, out=f"{figs_dir}/calibration.png")
    plot_lift_curve(y_test, y_proba, f"{figs_dir}/lift.png")
    plot_gain_curve(y_test, y_proba, f"{figs_dir}/gain.png")

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model("models/lgbm.pkl", "data/processed/features.pkl")
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
