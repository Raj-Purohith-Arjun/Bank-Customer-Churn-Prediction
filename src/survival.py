# src/survival.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


# kaplan meier
def fit_kaplan_meier(df, duration_col, event_col):
    if not HAS_LIFELINES:
        return None
    kmf = KaplanMeierFitter()
    kmf.fit(df[duration_col], df[event_col])
    return kmf


# cox model
def fit_cox(df, duration_col, event_col):
    if not HAS_LIFELINES:
        return None
    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col)
    return cph


# plot survival
def plot_km_curve(kmf, out="figs/km_survival.png"):
    if kmf is None:
        return
    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function()
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Survival Probability")
    plt.title("Kaplan-Meier Survival Curve")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# plot km segments
def plot_km_by_segment(df, duration_col, event_col, segment_col, out="figs/km_segments.png"):
    if not HAS_LIFELINES:
        print("lifelines not installed.")
        return
    
    plt.figure(figsize=(10, 6))
    for seg in df[segment_col].unique():
        mask = df[segment_col] == seg
        kmf = KaplanMeierFitter()
        kmf.fit(df.loc[mask, duration_col], df.loc[mask, event_col], label=str(seg))
        kmf.plot_survival_function()
    
    plt.xlabel("Tenure (Months)")
    plt.ylabel("Survival Probability")
    plt.title(f"Survival by {segment_col}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# plot cox coef
def plot_cox_coefficients(cph, out="figs/cox_coefficients.png"):
    if cph is None:
        return
    plt.figure(figsize=(10, 8))
    cph.plot()
    plt.title("Cox Hazard Ratios")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# survival analysis
def run_survival_analysis(data_path, figs_dir="figs"):
    os.makedirs(figs_dir, exist_ok=True)
    
    if not HAS_LIFELINES:
        print("lifelines not installed. Skip survival analysis.")
        return
    
    # load data
    df = pd.read_csv(data_path)
    
    # basic clean
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # use Tenure as duration
    if "Tenure" not in df.columns:
        print("No Tenure column. Skip survival.")
        return
    
    # kaplan meier
    kmf = fit_kaplan_meier(df, "Tenure", "Exited")
    plot_km_curve(kmf, f"{figs_dir}/km_survival.png")
    
    # by geography
    if "Geography" in df.columns:
        plot_km_by_segment(df, "Tenure", "Exited", "Geography", f"{figs_dir}/km_geography.png")
    
    # by gender
    if "Gender" in df.columns:
        plot_km_by_segment(df, "Tenure", "Exited", "Gender", f"{figs_dir}/km_gender.png")
    
    # cox model
    cox_cols = ["Tenure", "Exited", "Age", "Balance", "NumOfProducts", "IsActiveMember"]
    cox_cols = [c for c in cox_cols if c in df.columns]
    df_cox = df[cox_cols].dropna()
    cph = fit_cox(df_cox, "Tenure", "Exited")
    plot_cox_coefficients(cph, f"{figs_dir}/cox_coefficients.png")
    
    print("Survival analysis done.")


if __name__ == "__main__":
    run_survival_analysis("data/raw/bank_customers.csv")
