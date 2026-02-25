# tests/test_basic.py
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDataPrep:
    # test load
    def test_load_raw(self):
        from src.data_prep import load_raw
        df = load_raw("data/raw/bank_customers.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    # test clean
    def test_basic_clean(self):
        from src.data_prep import load_raw, basic_clean
        df = load_raw("data/raw/bank_customers.csv")
        df = basic_clean(df)
        assert "RowNumber" not in df.columns
        assert "CustomerId" not in df.columns

    # test split
    def test_split_data(self):
        from src.data_prep import load_raw, basic_clean, encode_categories, split_data
        df = load_raw("data/raw/bank_customers.csv")
        df = basic_clean(df)
        df = encode_categories(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
        assert len(X_train) > len(X_val)
        assert len(X_train) > len(X_test)


class TestFeatures:
    # test ratios
    def test_add_ratio_features(self):
        from src.features import add_ratio_features
        df = pd.DataFrame({
            "Balance": [1000, 2000],
            "EstimatedSalary": [50000, 100000],
            "NumOfProducts": [1, 2],
            "Tenure": [5, 10]
        })
        df = add_ratio_features(df)
        assert "balance_salary_ratio" in df.columns
        assert "products_per_tenure" in df.columns

    # test flags
    def test_add_flags(self):
        from src.features import add_flags
        df = pd.DataFrame({
            "Balance": [0, 1000],
            "NumOfProducts": [1, 2]
        })
        df = add_flags(df)
        assert "is_zero_balance" in df.columns
        assert df["is_zero_balance"].iloc[0] == 1


class TestTrain:
    # test build
    def test_build_logreg(self):
        from src.train import build_logreg
        pipe = build_logreg()
        assert pipe is not None

    # test build rf
    def test_build_rf(self):
        from src.train import build_rf
        pipe = build_rf()
        assert pipe is not None


class TestEvaluate:
    # test auc
    def test_compute_auc(self):
        from src.evaluate import compute_auc
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        auc = compute_auc(y_true, y_proba)
        assert auc == 1.0

    # test precision k
    def test_precision_at_k(self):
        from src.evaluate import precision_at_k
        y_true = np.array([0, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        prec = precision_at_k(y_true, y_proba, k=0.4)
        assert prec == 1.0


class TestMonitoring:
    # test psi
    def test_compute_psi(self):
        from src.monitoring import compute_psi
        np.random.seed(42)
        expected = np.random.normal(0, 1, 1000)
        np.random.seed(42)
        actual = np.random.normal(0.1, 1, 1000)  # small shift
        psi = compute_psi(expected, actual)
        assert psi >= 0  # psi is non-negative

    # test ks
    def test_ks_test(self):
        from src.monitoring import ks_test
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)
        stat, pval = ks_test(expected, actual)
        assert pval > 0.01  # same dist
