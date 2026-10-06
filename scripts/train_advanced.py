"""Fit fixed Phase 5 tree models with chronological, fold-local preprocessing."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import lightgbm
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from src.data.eda import file_manifest
from src.data.load_stats19 import load_collision_data, read_codebook
from src.data.splits import quarterly_folds, temporal_split
from src.evaluation.classification import evaluate_predictions, ordered_probabilities, select_baseline
from src.features.preprocessing import RAW_FEATURES, encode_target
from src.models.advanced import ADVANCED_NAMES, build_advanced, fit_advanced


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")


def predict_checked(pipeline, X):
    probabilities = ordered_probabilities(pipeline.predict_proba(X), pipeline.named_steps["model"].classes_)
    if not np.array_equal(pipeline.predict(X), probabilities.argmax(axis=1)):
        raise ValueError("Prediction does not match aligned probability argmax")
    return probabilities


def _train(root: Path, selected_names: tuple[str, ...]) -> dict:
    protocol = json.loads((root / "config/advanced_protocol.json").read_text(encoding="utf-8"))
    if protocol["models"] != list(ADVANCED_NAMES) or protocol["test_scoring"] or protocol["hyperparameter_search"]:
        raise ValueError("Advanced experiment protocol changed outside its declared scope")
    if protocol["primary_selection"] != "mean_macro_f1_on_2019_quarterly_cv":
        raise ValueError("Advanced model selection must use 2019 quarterly CV")
    if not selected_names or not set(selected_names).issubset(ADVANCED_NAMES):
        raise ValueError("Requested unknown advanced models")
    manifest = json.loads((root / "reports/eda/data_manifest.json").read_text(encoding="utf-8"))
    paths = [root / item["path"] for item in manifest["files"]]
    if file_manifest(paths, root) != manifest["files"]:
        raise ValueError("Raw dataset snapshot changed since EDA")
    guide = root / "config/stats19_data_guide_2025.xlsx"
    data = load_collision_data(root / "data/raw/stats19", years=(2019, 2020), guide_path=guide)
    parts = temporal_split(data, {"train": 2019, "validation": 2020})
    train, validation = parts["train"], parts["validation"]
    X_train, X_valid = train.loc[:, RAW_FEATURES], validation.loc[:, RAW_FEATURES]
    y_train, y_valid = encode_target(train["collision_severity"]), encode_target(validation["collision_severity"])
    folds = quarterly_folds(train)
    report = root / "reports/advanced"
    artifacts = root / "models/advanced_v1"
    private = root / "data/processed/advanced_v1"
    for directory in (report, artifacts, private):
        directory.mkdir(parents=True, exist_ok=True)
    previous_path = root / "reports/advanced/summary.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.exists() else {}
    reuse_names = set(ADVANCED_NAMES) - set(selected_names)
    if reuse_names and any(name not in previous.get("results", {}) for name in reuse_names):
        raise ValueError("Cannot reuse missing results; run all advanced models first")
    cv_results = {name: previous["cv_results"][name] for name in reuse_names}
    results = {name: previous["results"][name] for name in reuse_names}
    fitted = {name: joblib.load(artifacts / f"{name}.joblib") for name in reuse_names}
    with threadpool_limits(limits=protocol["thread_limit"]):
        for name in selected_names:
            cv_results[name] = []
            for number, (fit, valid) in enumerate(folds, start=1):
                print(f"CV {name}: fold {number}/3", flush=True)
                model = build_advanced(name, read_codebook(guide), protocol)
                started = perf_counter()
                fit_advanced(model, X_train.iloc[fit], y_train[fit])
                fit_seconds = perf_counter() - started
                print(f"  fold fit completed in {fit_seconds:.1f}s", flush=True)
                probabilities = predict_checked(model, X_train.iloc[valid])
                cv_results[name].append({"fold": number, "fit_rows": len(fit), "validation_rows": len(valid),
                                         "fit_seconds": fit_seconds,
                                         "metrics": evaluate_predictions(y_train[valid], probabilities)})
            print(f"Full 2019 training: {name}", flush=True)
            model = build_advanced(name, read_codebook(guide), protocol)
            started = perf_counter()
            fit_advanced(model, X_train, y_train)
            fit_seconds = perf_counter() - started
            print(f"  full fit completed in {fit_seconds:.1f}s", flush=True)
            valid_probabilities = predict_checked(model, X_valid)
            train_probabilities = predict_checked(model, X_train)
            np.save(private / f"{name}_validation_probabilities.npy", valid_probabilities, allow_pickle=False)
            target = artifacts / f"{name}.joblib"
            joblib.dump(model, target, compress=3)
            restored = joblib.load(target)
            if not np.allclose(predict_checked(restored, X_valid.iloc[:64]), valid_probabilities[:64], atol=1e-8):
                raise ValueError(f"Artifact round-trip mismatch: {name}")
            probe = X_valid.iloc[:min(128, len(X_valid))]
            started = perf_counter()
            restored.predict_proba(probe)
            latency_ms = (perf_counter() - started) * 1000
            results[name] = {"train": evaluate_predictions(y_train, train_probabilities),
                             "validation": evaluate_predictions(y_valid, valid_probabilities),
                             "fit_seconds": fit_seconds, "inference_batch_size": len(probe),
                             "inference_batch_ms": latency_ms, "artifact_bytes": target.stat().st_size,
                             "parameters": model.named_steps["model"].get_params()}
            fitted[name] = model
    selected = select_baseline(cv_results)
    joblib.dump(fitted[selected], artifacts / "selected.joblib", compress=3)
    np.save(private / "validation_targets.npy", y_valid, allow_pickle=False)
    cv_summary = {}
    for name, records in cv_results.items():
        values = [record["metrics"]["macro_f1"] for record in records]
        cv_summary[name] = {"macro_f1_mean": float(np.mean(values)), "macro_f1_std": float(np.std(values, ddof=1))}
    summary = {"phase": 5, "completed": True, "train_rows": len(train), "validation_rows": len(validation),
               "selected_advanced": selected, "selection_source": "2019 quarterly CV macro F1",
               "test_scored": False, "protocol": protocol, "cv_results": cv_results, "cv_summary": cv_summary,
               "results": results, "input_manifest": manifest["files"],
               "code_manifest": file_manifest([root / p for p in ["config/advanced_protocol.json",
                   "src/models/advanced.py", "src/evaluation/classification.py", "src/features/preprocessing.py",
                   "scripts/train_advanced.py"]], root),
               "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                               "numpy": np.__version__, "lightgbm": lightgbm.__version__,
                               "joblib": joblib.__version__, "thread_limit": protocol["thread_limit"]},
               "checks": {"snapshot_matches_eda": True, "fresh_pipeline_per_cv_fold": True,
                          "training_only_fit": True, "aligned_probabilities": True,
                          "artifact_round_trip": True, "no_test_scoring": True}}
    write_json(report / "summary.json", summary)
    write_json(report / "metrics.json", {k: {"train": v["train"], "validation": v["validation"]}
                                         for k, v in results.items()})
    rows = ["| Model | CV macro F1 | 2020 macro F1 | Accuracy | Fatal recall | Serious recall | Log-loss |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, result in results.items():
        m = result["validation"]
        rows.append(f"| {name} | {cv_summary[name]['macro_f1_mean']:.4f} | {m['macro_f1']:.4f} | "
                    f"{m['accuracy']:.4f} | {m['per_class']['Fatal']['recall']:.4f} | "
                    f"{m['per_class']['Serious']['recall']:.4f} | {m['log_loss']:.4f} |")
    table = "\n".join(rows)
    (report / "comparison.md").write_text(table + "\n", encoding="utf-8")
    (report / "phase5_findings.md").write_text(
        "# Phase 5 — Advanced ML\n\n" +
        "Fixed Random Forest and LightGBM experiments use the same raw-input, fold-fitted preprocessing. "
        "The 2019 temporal folds alone select the advanced model; 2020 is held out for validation. "
        "No sampling or parameter search was used. Class weighting changes probability calibration and the "
        "precision/recall trade-off. These models estimate severity conditional on a recorded injury collision, "
        "not collision occurrence. 2021 was not scored.\n\n" + table + "\n\n" +
        f"Selected by 2019 CV macro F1: `{selected}`. CV standard deviation is descriptive across three "
        "overlapping expanding windows, not a confidence interval. Full class metrics are in `metrics.json`; "
        "parameters, timing and checks are in `summary.json`. The selected artifact is `models/advanced_v1/selected.joblib`.\n\n"
        "Learning notes: Random Forest averages randomized trees; LightGBM builds trees sequentially to reduce "
        "residual error. Macro F1 weights severity classes equally. Class weights affect the training objective "
        "and can raise minority recall while increasing false positives.\n", encoding="utf-8")
    print(table, flush=True)
    print(f"Selected by 2019 CV: {selected}; 2021 test not scored", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--models", nargs="+", choices=ADVANCED_NAMES, default=list(ADVANCED_NAMES))
    arguments = parser.parse_args()
    _train(arguments.root.resolve(), tuple(arguments.models))


if __name__ == "__main__":
    main()
