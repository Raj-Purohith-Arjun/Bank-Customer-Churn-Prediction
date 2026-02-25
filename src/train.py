# src/train.py
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


# build logreg
def build_logreg():
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    return pipe


# build rf
def build_rf():
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    return pipe


# build lgbm
def build_lgbm():
    if not HAS_LGBM:
        return None
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1))
    ])
    return pipe


# build xgb
def build_xgb():
    if not HAS_XGB:
        return None
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric="logloss"))
    ])
    return pipe


# build catboost
def build_catboost():
    if not HAS_CAT:
        return None
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", CatBoostClassifier(n_estimators=500, learning_rate=0.05, random_state=42, verbose=0))
    ])
    return pipe


# build stacking
def build_stacking():
    estimators = []
    if HAS_LGBM:
        estimators.append(("lgbm", LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)))
    if HAS_XGB:
        estimators.append(("xgb", XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric="logloss")))
    estimators.append(("rf", RandomForestClassifier(n_estimators=100, random_state=42)))

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
        n_jobs=-1
    )
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", stack)
    ])
    return pipe


# train fit
def fit_model(pipe, X_train, y_train):
    pipe.fit(X_train, y_train)
    return pipe


# save model
def save_model(pipe, path):
    joblib.dump(pipe, path)


# load model
def load_model(path):
    return joblib.load(path)


# train all
def train_all(data_path, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    splits = pd.read_pickle(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = splits

    models = {
        "logreg": build_logreg(),
        "rf": build_rf(),
        "lgbm": build_lgbm(),
        "xgb": build_xgb(),
        "catboost": build_catboost(),
        "stacking": build_stacking()
    }

    results = {}
    for name, pipe in models.items():
        if pipe is None:
            continue
        print(f"Training {name}...")
        pipe = fit_model(pipe, X_train, y_train)
        save_model(pipe, f"{models_dir}/{name}.pkl")

        # evaluate
        y_pred = pipe.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        results[name] = auc
        print(f"  {name} AUC: {auc:.4f}")

    return results


if __name__ == "__main__":
    results = train_all("data/processed/features.pkl", "models")
    print("\nModel Results:")
    for name, auc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name}: {auc:.4f}")
