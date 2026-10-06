from __future__ import annotations

import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.features.preprocessing import make_model_pipeline


BASELINE_NAMES = ("dummy_prior", "logistic", "logistic_balanced")
SEED = 42


def build_baseline(name: str, codebook: dict, *, seed: int = SEED, logistic_params: dict | None = None):
    if name == "dummy_prior":
        estimator = DummyClassifier(strategy="prior", random_state=seed)
    elif name in {"logistic", "logistic_balanced"}:
        parameters = logistic_params if logistic_params is not None else {
            "C": 1.0, "solver": "lbfgs", "penalty": "l2", "tol": 1e-6, "max_iter": 2000}
        estimator = LogisticRegression(**parameters, random_state=seed,
                                       class_weight="balanced" if name == "logistic_balanced" else None)
    else:
        raise ValueError(f"Unknown baseline: {name}")
    return make_model_pipeline(estimator, codebook)


def fit_baseline(pipeline, X, y):
    if not np.array_equal(np.unique(y), [0, 1, 2]):
        raise ValueError("Baseline training requires all three severity classes")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        pipeline.fit(X, y)
    return pipeline
