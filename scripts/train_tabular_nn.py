"""Train a compact PyTorch classifier with nested chronological early stopping."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch

from src.data.eda import file_manifest
from src.data.load_stats19 import load_collision_data, read_codebook
from src.data.splits import quarterly_folds
from src.evaluation.classification import evaluate_predictions
from src.features.preprocessing import RAW_FEATURES, encode_target, make_preprocessor, recorded_timestamp
from src.models.tabular_nn import (TabularSeverityPredictor, dense_tensor, fit_early_stopping,
                                  fit_fixed_epochs)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _predict(model, matrix):
    with torch.no_grad():
        probabilities = torch.softmax(model(dense_tensor(matrix)), dim=1).cpu().numpy().astype(np.float64)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _inner_split(X, y, outer_fit, next_quarter):
    timestamps = recorded_timestamp(X.iloc[outer_fit])
    quarters = timestamps.dt.quarter.to_numpy()
    stop_quarter = next_quarter - 1
    fit_mask = quarters < stop_quarter
    stop_mask = quarters == stop_quarter
    fit = outer_fit[fit_mask]
    stop = outer_fit[stop_mask]
    if next_quarter == 2:
        q1 = outer_fit[quarters == 1]
        boundary = max(1, int(len(q1) * 0.8))
        fit, stop = q1[:boundary], q1[boundary:]
    if len(fit) == 0 or len(stop) == 0:
        raise ValueError(f"Insufficient data for inner early-stopping split before Q{next_quarter}")
    return fit, stop


def _plot_history(report: Path, fold_histories: list[dict]) -> None:
    fig, axes = plt.subplots(1, len(fold_histories), figsize=(15, 4.5), sharey=True)
    for ax, record in zip(axes, fold_histories):
        history = record["history"]
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="Train weighted loss")
        ax.plot(epochs, history["validation_loss"], label="Early-stop validation loss")
        ax.axvline(history["best_epoch"], color="#b34d4d", linestyle="--", label="Best epoch")
        ax.set(title=f"Outer validation Q{record['fold'] + 1}", xlabel="Epoch", ylabel="Cross-entropy")
    axes[-1].legend(fontsize=8)
    fig.suptitle("PyTorch MLP: nested chronological early stopping")
    fig.tight_layout()
    fig.savefig(report / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _train(root: Path) -> dict:
    config = json.loads((root / "config/deep_learning_protocol.json").read_text(encoding="utf-8"))
    if config["test_scoring"] or config["loss"] != "weighted_cross_entropy" or config["optimizer"] != "Adam":
        raise ValueError("Unexpected deep-learning protocol")
    manifest = json.loads((root / "reports/eda/data_manifest.json").read_text(encoding="utf-8"))
    if file_manifest([root / item["path"] for item in manifest["files"]], root) != manifest["files"]:
        raise ValueError("Raw dataset snapshot changed since EDA")
    guide = root / "config/stats19_data_guide_2025.xlsx"
    data = load_collision_data(root / "data/raw/stats19", years=(2019, 2020), guide_path=guide)
    from src.data.splits import temporal_split
    parts = temporal_split(data, {"train": 2019, "validation": 2020})
    train, valid = parts["train"], parts["validation"]
    X_train, X_valid = train.loc[:, RAW_FEATURES], valid.loc[:, RAW_FEATURES]
    y_train, y_valid = encode_target(train["collision_severity"]), encode_target(valid["collision_severity"])
    folds = quarterly_folds(train)
    codebook = read_codebook(guide)
    report, artifact_dir = root / "reports/deep_learning", root / "models/tabular_nn_v1"
    report.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cv_results, fold_histories, selected_epochs = [], [], []
    for fold_index, (fit_indices, valid_indices) in enumerate(folds):
        outer_quarter = fold_index + 2
        inner_fit, inner_stop = _inner_split(X_train, y_train, fit_indices, outer_quarter)
        preprocessor = make_preprocessor(codebook).fit(X_train.iloc[inner_fit])
        fit_matrix = preprocessor.transform(X_train.iloc[inner_fit])
        stop_matrix = preprocessor.transform(X_train.iloc[inner_stop])
        print(f"MLP early stopping for Q{outer_quarter} validation", flush=True)
        pilot, history = fit_early_stopping(fit_matrix, y_train[inner_fit], stop_matrix, y_train[inner_stop],
                                            config, seed=config["seed"] + fold_index)
        selected_epochs.append(history["best_epoch"])
        fold_histories.append({"fold": fold_index, "history": history,
                               "inner_fit_rows": len(inner_fit), "inner_validation_rows": len(inner_stop)})
        outer_preprocessor = make_preprocessor(codebook).fit(X_train.iloc[fit_indices])
        outer_matrix = outer_preprocessor.transform(X_train.iloc[fit_indices])
        outer_valid_matrix = outer_preprocessor.transform(X_train.iloc[valid_indices])
        refit = fit_fixed_epochs(outer_matrix, y_train[fit_indices], config,
                                 epochs=history["best_epoch"], seed=config["seed"] + fold_index)
        probabilities = _predict(refit, outer_valid_matrix)
        cv_results.append({"fold": fold_index + 1, "validation_quarter": outer_quarter,
                           "fit_rows": len(fit_indices), "validation_rows": len(valid_indices),
                           "inner_fit_rows": len(inner_fit), "early_stop_rows": len(inner_stop),
                           "best_epoch": history["best_epoch"],
                           "early_stop_history": history,
                           "metrics": evaluate_predictions(y_train[valid_indices], probabilities)})
    final_epochs = max(1, int(np.median(selected_epochs)))
    final_preprocessor = make_preprocessor(codebook).fit(X_train)
    final_matrix = final_preprocessor.transform(X_train)
    final_model = fit_fixed_epochs(final_matrix, y_train, config, epochs=final_epochs,
                                   seed=config["seed"] + 100)
    predictor = TabularSeverityPredictor(final_preprocessor, final_model)
    valid_probabilities = predictor.predict_proba(X_valid)
    train_probabilities = predictor.predict_proba(X_train)
    artifact = artifact_dir / "selected.joblib"
    joblib.dump(predictor, artifact, compress=3)
    restored = joblib.load(artifact)
    if not np.allclose(restored.predict_proba(X_valid.iloc[:64]), valid_probabilities[:64], atol=1e-7):
        raise ValueError("Neural artifact round-trip mismatch")
    np.save(report / "validation_probabilities.npy", valid_probabilities, allow_pickle=False)
    np.save(report / "validation_targets.npy", y_valid, allow_pickle=False)
    (report / "fold_histories.json").write_text(
        json.dumps(fold_histories, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")
    _plot_history(report, fold_histories)
    summary = {"phase": 6, "completed": True, "train_rows": len(train), "validation_rows": len(valid),
               "input_features": len(final_preprocessor.get_feature_names_out()),
               "output_classes": ["Fatal", "Serious", "Slight"], "selected_epochs_per_fold": selected_epochs,
               "final_epochs": final_epochs, "cv_results": cv_results,
               "validation": evaluate_predictions(y_valid, valid_probabilities),
               "train_in_sample": evaluate_predictions(y_train, train_probabilities),
               "test_scored": False, "protocol": config,
               "artifact": {"path": "models/tabular_nn_v1/selected.joblib", "bytes": artifact.stat().st_size},
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "numpy": np.__version__, "scikit_learn": sklearn.__version__, "device": "cpu"},
               "checks": {"fresh_preprocessor_per_fold": True, "early_stopping_inner_split": True,
                          "finite_probabilities": bool(np.isfinite(valid_probabilities).all()),
                          "artifact_round_trip": True, "test_not_scored": True}}
    (report / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")
    metrics = {"model": "PyTorch MLP", "cv_macro_f1": [row["metrics"]["macro_f1"] for row in cv_results],
               "cv_macro_f1_mean": float(np.mean([row["metrics"]["macro_f1"] for row in cv_results])),
               "validation_accuracy": summary["validation"]["accuracy"],
               "validation_macro_f1": summary["validation"]["macro_f1"],
               "validation_fatal_recall": summary["validation"]["per_class"]["Fatal"]["recall"],
               "validation_serious_recall": summary["validation"]["per_class"]["Serious"]["recall"]}
    (report / "comparison.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    report = root / "reports/deep_learning"
    report.mkdir(parents=True, exist_ok=True)
    (report / "run_status.json").write_text('{"status":"running"}', encoding="utf-8")
    try:
        _train(root)
    except Exception as error:
        (report / "run_status.json").write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2),
                                                 encoding="utf-8")
        raise
    (report / "run_status.json").write_text('{"status":"completed"}', encoding="utf-8")


if __name__ == "__main__":
    main()
