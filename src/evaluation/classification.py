from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, log_loss, precision_recall_fscore_support, roc_auc_score)

from src.features.preprocessing import TARGET_LABELS


LABELS = np.array([0, 1, 2])


def ordered_probabilities(probabilities, classes) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    if (classes.ndim != 1 or len(classes) == 0 or len(np.unique(classes)) != len(classes)
            or not np.isin(classes, LABELS).all()):
        raise ValueError("Invalid or duplicate model classes")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != len(classes):
        raise ValueError("Probability shape must match model class columns")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be finite and within [0,1]")
    if not np.allclose(values.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Probability rows must sum to one")
    values = values / values.sum(axis=1, keepdims=True)
    aligned = np.zeros((len(values), 3), dtype=float)
    for column, label in enumerate(classes):
        aligned[:, int(label)] = values[:, column]
    return aligned


def evaluate_predictions(y_true, probabilities, classes=(0, 1, 2)) -> dict:
    truth = np.asarray(y_true)
    values = ordered_probabilities(probabilities, classes)
    if (truth.ndim != 1 or len(truth) != len(values) or truth.dtype.kind not in "iu"
            or not np.isin(truth, LABELS).all()):
        raise ValueError("Nonempty aligned integer severity targets required")
    predicted = values.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=LABELS, zero_division=0)
    per_class = {}
    aucs, aps = [], []
    for index, label in enumerate(LABELS):
        binary = truth == label
        evaluable = binary.any() and not binary.all()
        auc = float(roc_auc_score(binary, values[:, index])) if evaluable else None
        ap = float(average_precision_score(binary, values[:, index])) if evaluable else None
        aucs.append(auc)
        aps.append(ap)
        count = int((predicted == label).sum())
        per_class[TARGET_LABELS[int(label)]] = {
            "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index]),
            "support": int(support[index]), "predicted_count": count,
            "precision_defined": count > 0, "recall_defined": support[index] > 0,
            "prevalence": float(binary.mean()), "roc_auc_ovr": auc, "average_precision_ovr": ap,
        }
    one_hot = np.eye(3)[truth]
    observed = support > 0
    return {"rows": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
            "balanced_accuracy": float(recall[observed].mean()),
            "macro_f1": float(f1.mean()), "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
            "log_loss": float(log_loss(truth, values, labels=LABELS)),
            "multiclass_brier": float(np.square(values - one_hot).sum(axis=1).mean()),
            "roc_auc_ovr_macro": float(np.mean(aucs)) if all(v is not None for v in aucs) else None,
            "average_precision_ovr_macro": float(np.mean(aps)) if all(v is not None for v in aps) else None,
            "labels": list(TARGET_LABELS.values()), "confusion_matrix": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
            "per_class": per_class}


def select_baseline(cv_results: dict[str, list[dict]]) -> str:
    if not cv_results or any(not folds for folds in cv_results.values()):
        raise ValueError("CV results must contain nonempty folds for each baseline")
    scores = {}
    for name, folds in cv_results.items():
        values = np.array([fold["metrics"]["macro_f1"] for fold in folds])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite CV selection scores")
        scores[name] = float(values.mean())
    return sorted(scores, key=lambda name: (-scores[name], name))[0]
