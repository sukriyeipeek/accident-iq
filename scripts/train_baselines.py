"""Fit reproducible Phase 4 baselines on raw development rows, never the test set."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from src.data.eda import file_manifest
from src.data.load_stats19 import load_collision_data, read_codebook
from src.data.splits import quarterly_folds, temporal_split
from src.evaluation.baseline_report import METRICS, metric_table, plot_reports, write_findings
from src.evaluation.classification import evaluate_predictions, ordered_probabilities, select_baseline
from src.features.preprocessing import RAW_FEATURES, encode_target, recorded_timestamp
from src.models.baselines import BASELINE_NAMES, build_baseline, fit_baseline


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
                               default=lambda item: item.item() if isinstance(item, np.generic) else TypeError(
                                   f"Object of type {type(item).__name__} is not JSON serializable")),
                    encoding="utf-8")


def pipeline_metadata(pipeline) -> dict:
    preprocessor = pipeline.named_steps["preprocessing"]
    columns = preprocessor.named_steps["columns"]
    speed = columns.named_transformers_["speed"]
    return {"fit_rows": int(speed.named_steps["scale"].n_samples_seen_),
            "speed_median_mph": float(speed.named_steps["impute"].statistics_[0]),
            "feature_count": len(preprocessor.get_feature_names_out()),
            "feature_names": list(preprocessor.get_feature_names_out()),
            "classes": pipeline.named_steps["model"].classes_.tolist(),
            "iterations": getattr(pipeline.named_steps["model"], "n_iter_", np.array([])).tolist()}


def predict_checked(pipeline, X):
    state = joblib.hash(pipeline)
    probability = ordered_probabilities(pipeline.predict_proba(X), pipeline.named_steps["model"].classes_)
    if joblib.hash(pipeline) != state:
        raise ValueError("Prediction changed fitted pipeline state")
    if not np.array_equal(pipeline.predict(X), probability.argmax(axis=1)):
        raise ValueError("Model decision does not match aligned probability argmax")
    return probability


def save_pipeline(pipeline, path: Path, probe) -> int:
    partial = path.with_suffix(path.suffix + ".part")
    joblib.dump(pipeline, partial, compress=3)
    partial.replace(path)
    restored = joblib.load(path)
    if not np.array_equal(predict_checked(restored, probe), predict_checked(pipeline, probe)):
        raise ValueError("Saved classifier round-trip probability mismatch")
    return path.stat().st_size


def _run_baselines(root: Path) -> dict:
    protocol = json.loads((root / "config/baseline_protocol.json").read_text(encoding="utf-8"))
    if (protocol["train_year"] != 2019 or protocol["validation_year"] != 2020 or protocol["test_scoring"]
            or protocol["models"] != list(BASELINE_NAMES) or protocol["hyperparameter_search"]
            or protocol["primary_selection"] != "mean_macro_f1_on_2019_quarterly_cv"
            or protocol["tie_break"] != "lexicographic_model_name" or protocol["sampling"] != "none"
            or protocol["decision_rule"] != "multiclass_argmax"):
        raise ValueError("Protocol changed beyond the Phase 4 baseline scope")
    source = json.loads((root / "reports/eda/data_manifest.json").read_text(encoding="utf-8"))
    paths = [root / item["path"] for item in source["files"]]
    if file_manifest(paths, root) != source["files"]:
        raise ValueError("Dataset changed since EDA; review snapshot before training")
    guide = root / "config/stats19_data_guide_2025.xlsx"
    codebook = read_codebook(guide)
    development = load_collision_data(root / "data/raw/stats19", years=(2019, 2020), guide_path=guide)
    partitions = temporal_split(development, {"train": 2019, "validation": 2020})
    train, valid = partitions["train"], partitions["validation"]
    X_train, X_valid = train.loc[:, RAW_FEATURES], valid.loc[:, RAW_FEATURES]
    y_train, y_valid = encode_target(train["collision_severity"]), encode_target(valid["collision_severity"])
    folds = quarterly_folds(train)
    report = root / "reports/baselines"
    artifact_dir = root / "models/baseline_v1"
    private = root / "data/processed/baseline_v1"
    for directory in (report, artifact_dir, private):
        directory.mkdir(parents=True, exist_ok=True)
    cv_results, results, probabilities = {}, {}, {}
    fitted = {}
    with threadpool_limits(limits=protocol["thread_limit"]):
        for name in BASELINE_NAMES:
            cv_results[name] = []
            for number, (fit, validation) in enumerate(folds, start=1):
                print(f"CV {name}: fold {number}/3", flush=True)
                pipeline = build_baseline(name, codebook, seed=protocol["seed"], logistic_params=protocol["logistic"])
                start = perf_counter()
                fit_baseline(pipeline, X_train.iloc[fit], y_train[fit])
                seconds = perf_counter() - start
                metadata = pipeline_metadata(pipeline)
                if metadata["fit_rows"] != len(fit):
                    raise ValueError("Fold preprocessing fit count mismatch")
                probability = predict_checked(pipeline, X_train.iloc[validation])
                cv_results[name].append({"fold": number, "fit_rows": len(fit), "validation_rows": len(validation),
                                         "fit_until": recorded_timestamp(X_train.iloc[fit]).max().isoformat(),
                                         "validation_from": recorded_timestamp(X_train.iloc[validation]).min().isoformat(),
                                         "fit_seconds": seconds, "pipeline": metadata,
                                         "metrics": evaluate_predictions(y_train[validation], probability)})
            print(f"Full 2019 training: {name}", flush=True)
            pipeline = build_baseline(name, codebook, seed=protocol["seed"], logistic_params=protocol["logistic"])
            start = perf_counter()
            fit_baseline(pipeline, X_train, y_train)
            fit_seconds = perf_counter() - start
            metadata = pipeline_metadata(pipeline)
            if metadata["fit_rows"] != len(train):
                raise ValueError("Full preprocessing fit count mismatch")
            train_probability = predict_checked(pipeline, X_train)
            probability = predict_checked(pipeline, X_valid)
            probabilities[name] = probability
            np.save(private / f"{name}_validation_probabilities.npy", probability, allow_pickle=False)
            probe = X_valid.iloc[:min(128, len(valid))]
            pipeline.predict_proba(probe)
            times = []
            for _ in range(5):
                start = perf_counter()
                pipeline.predict_proba(probe)
                times.append(perf_counter() - start)
            bytes_written = save_pipeline(pipeline, artifact_dir / f"{name}.joblib", probe)
            counts = np.bincount(y_train, minlength=3)
            results[name] = {"train": evaluate_predictions(y_train, train_probability),
                             "validation": evaluate_predictions(y_valid, probability),
                             "fit_seconds": fit_seconds, "inference_batch_size": len(probe),
                             "inference_batch_median_ms": float(np.median(times) * 1000),
                             "artifact_bytes": bytes_written, "pipeline": metadata,
                             "training_class_weights": {str(label): float(len(y_train) / (3 * n)) if name == "logistic_balanced" else 1.0
                                                        for label, n in enumerate(counts)},
                             "parameters": pipeline.named_steps["model"].get_params()}
            fitted[name] = pipeline
    selected = select_baseline(cv_results)
    save_pipeline(fitted[selected], artifact_dir / "selected.joblib", X_valid.iloc[:32])
    np.save(private / "validation_targets.npy", y_valid, allow_pickle=False)
    valid[["collision_index"]].assign(row_position=np.arange(len(valid))).to_csv(private / "validation_rows.csv", index=False)
    cv_summary = {}
    for name, records in cv_results.items():
        cv_summary[name] = {}
        for metric in METRICS:
            scores = [record["metrics"][metric] for record in records]
            cv_summary[name][f"{metric}_mean"] = float(np.mean(scores)) if all(v is not None for v in scores) else None
            cv_summary[name][f"{metric}_std"] = float(np.std(scores, ddof=1)) if all(v is not None for v in scores) else None
    summary = {"phase": 4, "completed": True, "train_rows": len(train), "validation_rows": len(valid),
               "selected_baseline": selected, "selection_source": "2019 quarterly CV macro F1 only",
               "test_scored": False, "test_loaded_for_training_or_scoring": False,
               "protocol": protocol, "cv_results": cv_results, "cv_summary": cv_summary, "results": results,
               "input_manifest": source["files"],
               "code_manifest": file_manifest([root / p for p in ["config/baseline_protocol.json", "src/features/preprocessing.py",
                   "src/data/splits.py", "src/models/baselines.py", "src/evaluation/classification.py",
                   "src/evaluation/baseline_report.py", "scripts/train_baselines.py"]], root),
               "artifact_manifest": file_manifest(sorted(artifact_dir.glob("*.joblib")) + sorted(private.glob("*")), root),
               "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                               "numpy": np.__version__, "joblib": joblib.__version__, "thread_limit": protocol["thread_limit"]},
               "checks": {"snapshot_matches_eda": True, "fresh_pipeline_per_cv_fold": True,
                          "training_only_fit_counts": True, "predict_does_not_mutate_fit": True,
                          "aligned_finite_probabilities": True, "no_convergence_warnings": True,
                          "classifier_serialization_round_trip": True, "no_test_scoring": True}}
    write_json(report / "summary.json", summary)
    write_json(report / "metrics.json", {name: {"train": result["train"], "validation": result["validation"]}
                                         for name, result in results.items()})
    write_json(report / "feature_names.json", results[selected]["pipeline"]["feature_names"])
    (report / "comparison.md").write_text(metric_table(results), encoding="utf-8")
    plot_reports(report, summary, y_valid, probabilities)
    write_findings(report, summary)
    print(metric_table(results), flush=True)
    print(f"Selected by 2019 CV: {selected}; 2021 test not scored", flush=True)
    return summary


def run_baselines(root: Path) -> dict:
    root = Path(root).resolve()
    report = root / "reports/baselines"
    report.mkdir(parents=True, exist_ok=True)
    write_json(report / "run_status.json", {"status": "running"})
    try:
        result = _run_baselines(root)
    except Exception as error:
        write_json(report / "run_status.json", {"status": "failed", "error_type": type(error).__name__, "message": str(error)})
        raise
    write_json(report / "run_status.json", {"status": "completed"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    run_baselines(parser.parse_args().root)


if __name__ == "__main__":
    main()
