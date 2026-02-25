# src/monitoring.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# compute drift
def compute_psi(expected, actual, bins=10):
    # population stability
    def _get_buckets(arr, bins):
        cut = pd.cut(arr, bins=bins, duplicates="drop")
        counts = cut.value_counts()
        return (counts / counts.sum()).sort_index()
    
    exp_pct = _get_buckets(expected, bins)
    act_pct = _get_buckets(actual, bins)
    
    # align bins
    all_bins = exp_pct.index.union(act_pct.index)
    exp_pct = exp_pct.reindex(all_bins, fill_value=0.001)
    act_pct = act_pct.reindex(all_bins, fill_value=0.001)
    
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return psi


# ks test
def ks_test(expected, actual):
    stat, pval = stats.ks_2samp(expected, actual)
    return stat, pval


# feature drift
def compute_feature_drift(df_expected, df_actual):
    results = {}
    common_cols = list(set(df_expected.columns) & set(df_actual.columns))
    
    for col in common_cols:
        if df_expected[col].dtype in ["float64", "int64"]:
            psi = compute_psi(df_expected[col].dropna(), df_actual[col].dropna())
            ks_stat, ks_pval = ks_test(df_expected[col].dropna(), df_actual[col].dropna())
            results[col] = {
                "psi": psi,
                "ks_stat": ks_stat,
                "ks_pval": ks_pval,
                "drift_alert": psi > 0.2 or ks_pval < 0.01
            }
    return results


# plot drift
def plot_drift_comparison(df_expected, df_actual, col, out="figs/drift.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # histogram
    axes[0].hist(df_expected[col].dropna(), bins=30, alpha=0.5, label="Expected", density=True)
    axes[0].hist(df_actual[col].dropna(), bins=30, alpha=0.5, label="Actual", density=True)
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].set_title(f"{col} Distribution")
    
    # cumulative
    axes[1].hist(df_expected[col].dropna(), bins=30, alpha=0.5, label="Expected", density=True, cumulative=True)
    axes[1].hist(df_actual[col].dropna(), bins=30, alpha=0.5, label="Actual", density=True, cumulative=True)
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("Cumulative Density")
    axes[1].legend()
    axes[1].set_title(f"{col} Cumulative")
    
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# auc drift
def track_auc_drift(auc_history, threshold=0.05):
    if len(auc_history) < 2:
        return False
    recent = np.mean(auc_history[-3:])
    baseline = np.mean(auc_history[:3])
    drift = abs(recent - baseline)
    return drift > threshold


# alert logic
def generate_alerts(drift_results, auc_drift=False):
    alerts = []
    for col, metrics in drift_results.items():
        if metrics["drift_alert"]:
            alerts.append(f"DRIFT ALERT: {col} (PSI={metrics['psi']:.3f})")
    if auc_drift:
        alerts.append("AUC DRIFT ALERT: Model performance degraded")
    return alerts


# monitor report
def run_monitoring(expected_path, actual_path, figs_dir="figs"):
    os.makedirs(figs_dir, exist_ok=True)
    
    df_expected = pd.read_csv(expected_path)
    df_actual = pd.read_csv(actual_path)
    
    # basic clean
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df_expected = df_expected.drop(columns=[c for c in drop_cols if c in df_expected.columns])
    df_actual = df_actual.drop(columns=[c for c in drop_cols if c in df_actual.columns])
    
    # compute drift
    drift_results = compute_feature_drift(df_expected, df_actual)
    
    # plot sample
    for col in ["Age", "Balance", "CreditScore"]:
        if col in df_expected.columns and col in df_actual.columns:
            plot_drift_comparison(df_expected, df_actual, col, f"{figs_dir}/drift_{col}.png")
    
    # alerts
    alerts = generate_alerts(drift_results)
    
    print("Drift Analysis:")
    for col, metrics in drift_results.items():
        status = "⚠️ DRIFT" if metrics["drift_alert"] else "✓ OK"
        print(f"  {col}: PSI={metrics['psi']:.4f} {status}")
    
    if alerts:
        print("\nAlerts:")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("\nNo drift alerts.")
    
    return drift_results


if __name__ == "__main__":
    # self test
    run_monitoring("data/raw/bank_customers.csv", "data/raw/bank_customers.csv")
