# src/features.py
import pandas as pd
import numpy as np


# add ratios
def add_ratio_features(df):
    # balance ratio
    if "Balance" in df.columns and "EstimatedSalary" in df.columns:
        df["balance_salary_ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    # products ratio
    if "NumOfProducts" in df.columns and "Tenure" in df.columns:
        df["products_per_tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    return df


# add interactions
def add_interactions(df):
    # age credit
    if "Age" in df.columns and "CreditScore" in df.columns:
        df["age_credit_interaction"] = df["Age"] * df["CreditScore"]
    # balance products
    if "Balance" in df.columns and "NumOfProducts" in df.columns:
        df["balance_products"] = df["Balance"] * df["NumOfProducts"]
    return df


# add segments
def add_segments(df):
    # age segment
    if "Age" in df.columns:
        df["age_segment"] = pd.cut(
            df["Age"],
            bins=[0, 30, 45, 60, 100],
            labels=["young", "middle", "senior", "elderly"]
        )
        df = pd.get_dummies(df, columns=["age_segment"], prefix="age_seg")
    # balance segment
    if "Balance" in df.columns:
        df["balance_segment"] = pd.cut(
            df["Balance"],
            bins=[-1, 0, 50000, 100000, np.inf],
            labels=["zero", "low", "medium", "high"]
        )
        df = pd.get_dummies(df, columns=["balance_segment"], prefix="bal_seg")
    return df


# add flags
def add_flags(df):
    # zero balance
    if "Balance" in df.columns:
        df["is_zero_balance"] = (df["Balance"] == 0).astype(int)
    # single product
    if "NumOfProducts" in df.columns:
        df["is_single_product"] = (df["NumOfProducts"] == 1).astype(int)
    return df


# build features
def build_features(X_train, X_val, X_test):
    for transform in [add_ratio_features, add_interactions, add_segments, add_flags]:
        X_train = transform(X_train.copy())
        X_val = transform(X_val.copy())
        X_test = transform(X_test.copy())
    return X_train, X_val, X_test


if __name__ == "__main__":
    splits = pd.read_pickle("data/processed/splits.pkl")
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    X_train, X_val, X_test = build_features(X_train, X_val, X_test)
    pd.to_pickle((X_train, X_val, X_test, y_train, y_val, y_test), "data/processed/features.pkl")
    print("Features built.")
