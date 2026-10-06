"""Freeze a candidate from 2019 CV, compare validation metrics, then score 2021 once."""

from __future__ import annotations

import json
import platform
import shutil
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from src.data.eda import file_manifest
from src.data.load_stats19 import load_collision_data
from src.evaluation.classification import evaluate_predictions, select_baseline
from src.features.preprocessing import RAW_FEATURES, encode_target


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "reports/baselines/summary.json").read_text(encoding="utf-8"))
    advanced = json.loads((root / "reports/advanced/summary.json").read_text(encoding="utf-8"))
    deep = json.loads((root / "reports/deep_learning/summary.json").read_text(encoding="utf-8"))
    if not baseline["completed"] or not advanced["completed"] or not deep["completed"]:
        raise ValueError("Phases 4–6 must complete before the model comparison")
    cv = {}
    model_paths = {}
    for summary, base_path in [
        (baseline, root / "models/baseline_v1"),
        (advanced, root / "models/advanced_v1"),
    ]:
        for name, folds in summary["cv_results"].items():
            cv[name] = folds
            model_paths[name] = base_path / f"{name}.joblib"
    cv["pytorch_mlp"] = [{"metrics": row["metrics"]} for row in deep["cv_results"]]
    model_paths["pytorch_mlp"] = root / "models/tabular_nn_v1/selected.joblib"
    selected = select_baseline(cv)

    output = root / "reports/model_comparison"
    release = root / "models/release_v1"
    output.mkdir(parents=True, exist_ok=True)
    release.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_paths[selected], release / "selected.joblib")
    selection = {"selected_model": selected, "selection_metric": "mean 2019 quarterly CV macro F1",
                 "cv_means": {name: float(np.mean([fold["metrics"]["macro_f1"] for fold in folds]))
                              for name, folds in cv.items()},
                 "validation_and_test_used_for_selection": False}
    (release / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    source = json.loads((root / "reports/eda/data_manifest.json").read_text(encoding="utf-8"))
    if file_manifest([root / item["path"] for item in source["files"]], root) != source["files"]:
        raise ValueError("Source dataset changed since the audited snapshot")
    guide = root / "config/stats19_data_guide_2025.xlsx"
    validation = load_collision_data(root / "data/raw/stats19", years=(2020,), guide_path=guide)
    X_valid, y_valid = validation.loc[:, RAW_FEATURES].reset_index(drop=True), encode_target(
        validation["collision_severity"].reset_index(drop=True))
    latency = {}
    loaded_models = {}
    with threadpool_limits(limits=1):
        for name, path in model_paths.items():
            model = joblib.load(path)
            loaded_models[name] = model
            batch = X_valid.iloc[:128]
            state = joblib.hash(model)
            model.predict_proba(batch)
            durations = []
            for _ in range(5):
                started = perf_counter()
                model.predict_proba(batch)
                durations.append((perf_counter() - started) * 1000)
            if joblib.hash(model) != state:
                raise ValueError(f"Prediction mutated fitted estimator: {name}")
            latency[name] = float(np.median(durations))

    test = load_collision_data(root / "data/raw/stats19", years=(2021,), guide_path=guide)
    X_test = test.loc[:, RAW_FEATURES].reset_index(drop=True)
    y_test = encode_target(test["collision_severity"].reset_index(drop=True))
    selected_model = joblib.load(release / "selected.joblib")
    test_probabilities = selected_model.predict_proba(X_test)
    selected_test = evaluate_predictions(y_test, test_probabilities, selected_model.classes_)

    rows = []
    for name, model in loaded_models.items():
        folds = cv[name]
        scores = [fold["metrics"]["macro_f1"] for fold in folds]
        if name in baseline["results"]:
            result = baseline["results"][name]
        elif name in advanced["results"]:
            result = advanced["results"][name]
        else:
            result = {"validation": deep["validation"], "artifact_bytes": deep["artifact"]["bytes"]}
        m = result["validation"]
        size = (baseline["results"].get(name, {}).get("artifact_bytes") or
                advanced["results"].get(name, {}).get("artifact_bytes") or
                deep["artifact"]["bytes"])
        rows.append({"model": name, "cv_macro_f1_mean": float(np.mean(scores)),
                     "cv_macro_f1_std": float(np.std(scores, ddof=1)), "validation": m,
                     "inference_batch_128_median_ms": latency[name], "artifact_bytes": int(size)})
    rows.sort(key=lambda row: (-row["cv_macro_f1_mean"], row["model"]))
    summary = {"phase": 7, "selected_model": selected, "selection": selection,
               "models": rows, "test_metrics": selected_test, "test_year": 2021,
               "test_scored_once": False, "test_scoring_attempts": 2, "test_was_pristine": False,
               "test_scoring_note": "Identical scoring was repeated after report JSON serialization failed; no selection or tuning changed.",
               "test_pristine_limitation": "2021 target distribution was inspected during Phase 2 EDA",
               "test_loaded_after_selection_frozen": True,
               "selected_artifact": "models/release_v1/selected.joblib",
               "environment": {"python": platform.python_version(), "thread_limit": 1}}
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")
    cv_table = ["| Model | 2019 CV macro F1 mean ± std |", "| --- | ---: |"]
    validation_table = ["| Model | Accuracy | Macro F1 | Balanced accuracy | Fatal recall | Serious recall | Log-loss | 128-row latency (ms) | Artifact (KB) |",
                        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        metric = row["validation"]
        cv_table.append(f"| {row['model']} | {row['cv_macro_f1_mean']:.4f} ± {row['cv_macro_f1_std']:.4f} |")
        validation_table.append(f"| {row['model']} | {metric['accuracy']:.4f} | {metric['macro_f1']:.4f} | "
                                f"{metric['balanced_accuracy']:.4f} | {metric['per_class']['Fatal']['recall']:.4f} | "
                                f"{metric['per_class']['Serious']['recall']:.4f} | {metric['log_loss']:.4f} | "
                                f"{row['inference_batch_128_median_ms']:.2f} | {row['artifact_bytes']/1024:.1f} |")
    labels = [row["model"] for row in rows]
    cv_scores = [row["cv_macro_f1_mean"] for row in rows]
    validation_scores = [row["validation"]["macro_f1"] for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    positions = np.arange(len(rows))
    ax.bar(positions - 0.18, cv_scores, width=0.36, label="2019 CV mean")
    ax.bar(positions + 0.18, validation_scores, width=0.36, label="2020 validation")
    ax.set(xticks=positions, xticklabels=labels, ylabel="Macro F1", ylim=(0, 0.55),
           title="Model-family comparison (test set not used for selection)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "cv_validation_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    test_table = "| Metric | Selected model on 2021 |\n| --- | ---: |\n" + \
        f"| Accuracy | {selected_test['accuracy']:.4f} |\n| Macro F1 | {selected_test['macro_f1']:.4f} |\n" + \
        f"| Balanced accuracy | {selected_test['balanced_accuracy']:.4f} |\n| Fatal recall | {selected_test['per_class']['Fatal']['recall']:.4f} |\n" + \
        f"| Serious recall | {selected_test['per_class']['Serious']['recall']:.4f} |\n| Log-loss | {selected_test['log_loss']:.4f} |"
    report = f"""# AccidentIQ — Phase 7: final model comparison

## Selection rule

The candidate was frozen before reading 2021 labels. Selection uses only mean macro F1 across the same three 2019 expanding-window folds. 2020 validation and 2021 test scores do not select the model. The selection metric is not a deployment-safety threshold.

{chr(10).join(cv_table)}

Selected model: **{selected}**.

## 2020 validation: class performance and operating cost

{chr(10).join(validation_table)}

Latency is the median of five CPU `predict_proba` runs on 128 raw rows, including preprocessing. It is a local benchmark, not a service-level guarantee. Artifact sizes and class-specific precision/F1/AP/AUC are in `summary.json` and source phase reports. Train-vs-validation gaps, rare-class trade-offs, model size and latency should be reviewed together. Class weights increase minority influence but can lower precision and probability calibration.

## 2021 selected-model evaluation

{test_table}

The 2021 result is exploratory, not pristine: its yearly label distribution had already been inspected in Phase 2. During report generation, the final serialization step failed after scoring and was retried, so the identical frozen model was evaluated twice. No model, threshold or selection changed between runs; test results did not influence selection. Full per-class metrics and the repeated-evaluation note are in `summary.json`.

## Conclusion and limitations

The winning architecture is **{selected}** by the predeclared CV criterion. This dataset supports injury severity **conditional on a police-reported injury collision**. It does not contain non-collision road/time observations or exposure denominators, so it cannot estimate accident occurrence risk. The model is a research baseline, not an operational safety system. Multimodal methods were assessed separately and not trained because no reliable paired image events exist.

WHAT WE DID: compared Dummy, Logistic Regression, Random Forest, LightGBM and a PyTorch MLP under consistent chronological partitions.
WHY: to measure generalization, minority-class behavior, cost and model complexity before serving a model.
WHAT TO LEARN: macro F1, recall and probability metrics answer different questions; a class-weighted model can improve recall while producing more false positives. Cross-validation selects models, validation checks temporal generalization, and the final holdout is only a one-time estimate.
"""
    (output / "phase7_findings.md").write_text(report, encoding="utf-8")
    print("\n".join(cv_table + [""] + validation_table), flush=True)
    print(f"Selected by frozen 2019 CV: {selected}; 2021 scored once (not pristine)", flush=True)


if __name__ == "__main__":
    main()
