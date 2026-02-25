# src/data_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split


# load data
def load_raw(path):
    df = pd.read_csv(path)
    return df


# clean data
def basic_clean(df):
    # drop ids
    drop_cols = ["RowNumber", "CustomerId", "Surname"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # fill na
    df = df.fillna(df.median(numeric_only=True))
    # drop duplicates
    df = df.drop_duplicates()
    return df


# encode categories
def encode_categories(df):
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


# split data
def split_data(df, target="Exited", test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    # train val split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size * 1.25, random_state=random_state, stratify=y
    )
    # val test split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# main prep
def prepare_data(raw_path, processed_path):
    df = load_raw(raw_path)
    df = basic_clean(df)
    df = encode_categories(df)
    splits = split_data(df)
    pd.to_pickle(splits, processed_path)
    return splits


if __name__ == "__main__":
    import os
    os.makedirs("data/processed", exist_ok=True)
    prepare_data("data/raw/bank_customers.csv", "data/processed/splits.pkl")
    print("Data prepared.")
