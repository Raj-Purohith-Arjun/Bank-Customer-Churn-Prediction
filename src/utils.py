# src/utils.py
import os
import json
import joblib
import numpy as np
import pandas as pd


# save pickle
def save_pickle(obj, path):
    joblib.dump(obj, path)


# load pickle
def load_pickle(path):
    return joblib.load(path)


# save json
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# load json
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ensure dir
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# numpy encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# set seed
def set_seed(seed=42):
    np.random.seed(seed)
