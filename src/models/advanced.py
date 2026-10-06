from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

from src.features.preprocessing import make_model_pipeline, make_preprocessor


ADVANCED_NAMES = ("random_forest_balanced", "lightgbm", "lightgbm_balanced")


def named_dense_frame(matrix):
    values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return pd.DataFrame(values, columns=[f"Column_{index}" for index in range(values.shape[1])])


def build_advanced(name: str, codebook: dict, protocol: dict):
    seed = protocol["seed"]
    if name == "random_forest_balanced":
        params = {**protocol["random_forest"], "random_state": seed, "n_jobs": protocol["thread_limit"]}
        estimator = RandomForestClassifier(**params)
    elif name in {"lightgbm", "lightgbm_balanced"}:
        params = {**protocol["lightgbm"], "random_state": seed, "n_jobs": protocol["thread_limit"]}
        params["class_weight"] = "balanced" if name == "lightgbm_balanced" else None
        estimator = LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown advanced model: {name}")
    if name.startswith("lightgbm"):
        return Pipeline([("preprocessing", make_preprocessor(codebook)),
                         ("named_dense_frame", FunctionTransformer(named_dense_frame, validate=False)),
                         ("model", estimator)])
    return make_model_pipeline(estimator, codebook)


def fit_advanced(pipeline, X, y):
    pipeline.fit(X, y)
    return pipeline
