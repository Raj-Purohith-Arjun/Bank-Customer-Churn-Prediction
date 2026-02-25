# src/explain.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# load model
def load_model(path):
    return joblib.load(path)


# get explainer
def get_explainer(model, X_sample):
    if not HAS_SHAP:
        return None, None
    # get classifier
    clf = model.named_steps["clf"]
    # preprocess
    X_processed = model.named_steps["imp"].transform(X_sample)
    X_processed = model.named_steps["scaler"].transform(X_processed)
    # create explainer
    explainer = shap.Explainer(clf, X_processed)
    return explainer, X_processed


# shap summary
def plot_shap_summary(model, X_sample, feature_names, out="figs/shap_summary.png"):
    if not HAS_SHAP:
        print("SHAP not installed.")
        return
    
    explainer, X_processed = get_explainer(model, X_sample)
    if explainer is None:
        return
    
    shap_values = explainer(X_processed)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_processed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# shap waterfall
def plot_shap_waterfall(model, X_sample, idx, feature_names, out="figs/shap_waterfall.png"):
    if not HAS_SHAP:
        print("SHAP not installed.")
        return
    
    explainer, X_processed = get_explainer(model, X_sample)
    if explainer is None:
        return
    
    shap_values = explainer(X_processed)
    
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


# feature importance
def plot_feature_importance(model, feature_names, out="figs/feature_importance.png"):
    clf = model.named_steps["clf"]
    
    # get importance
    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_[0])
    else:
        print("No feature importance available.")
        return
    
    # sort
    idx = np.argsort(importance)[-20:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(idx)), importance[idx], color="steelblue")
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.xlabel("Importance")
    plt.title("Feature Importance (Top 20)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# explain model
def explain_model(model_path, data_path, figs_dir="figs", n_samples=500):
    os.makedirs(figs_dir, exist_ok=True)
    
    model = load_model(model_path)
    splits = pd.read_pickle(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    
    # sample
    X_sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=42)
    feature_names = list(X_sample.columns)
    
    # feature importance
    plot_feature_importance(model, feature_names, f"{figs_dir}/feature_importance.png")
    
    # shap plots
    if HAS_SHAP:
        plot_shap_summary(model, X_sample, feature_names, f"{figs_dir}/shap_summary.png")
        plot_shap_waterfall(model, X_sample, 0, feature_names, f"{figs_dir}/shap_waterfall_0.png")
        plot_shap_waterfall(model, X_sample, 1, feature_names, f"{figs_dir}/shap_waterfall_1.png")
    
    print("Explanations generated.")


if __name__ == "__main__":
    explain_model("models/lgbm.pkl", "data/processed/features.pkl")
