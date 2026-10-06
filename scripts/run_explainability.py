"""Generate global permutation and local sensitivity explanations for the frozen release model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.data.load_stats19 import load_collision_data, read_codebook
from src.evaluation.explainability import build_reference_row, local_sensitivity
from src.features.preprocessing import RAW_FEATURES, encode_target


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    selected = json.loads((root / "models/release_v1/selection.json").read_text(encoding="utf-8"))
    model = joblib.load(root / "models/release_v1/selected.joblib")
    guide = root / "config/stats19_data_guide_2025.xlsx"
    training = load_collision_data(root / "data/raw/stats19", years=(2019,), guide_path=guide)
    validation = load_collision_data(root / "data/raw/stats19", years=(2020,), guide_path=guide)
    X_train = training.loc[:, RAW_FEATURES]
    X_valid = validation.loc[:, RAW_FEATURES]
    y_valid = encode_target(validation["collision_severity"])
    sample_idx, _ = train_test_split(np.arange(len(y_valid)), train_size=2500,
                                     random_state=42, stratify=y_valid)
    X_sample = X_valid.iloc[sample_idx].reset_index(drop=True)
    y_sample = y_valid[sample_idx]
    scores = permutation_importance(model, X_sample, y_sample, scoring="f1_macro", n_repeats=3,
                                    random_state=42, n_jobs=1)
    table = pd.DataFrame({"feature": RAW_FEATURES, "macro_f1_drop_mean": scores.importances_mean,
                          "macro_f1_drop_std": scores.importances_std}).sort_values(
                              "macro_f1_drop_mean", ascending=False)
    report = root / "reports/explainability"
    report.mkdir(parents=True, exist_ok=True)
    table.to_csv(report / "permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ranked = table.sort_values("macro_f1_drop_mean")
    ax.barh(ranked["feature"], ranked["macro_f1_drop_mean"], xerr=ranked["macro_f1_drop_std"], color="#3b7c87")
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set(xlabel="Macro F1 decrease after permutation", title="2020 validation sample: grouped raw-input importance")
    fig.tight_layout()
    fig.savefig(report / "permutation_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    reference = build_reference_row(X_train)
    (root / "config/explanation_reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    example_index = int(sample_idx[0])
    local = local_sensitivity(model, X_valid.iloc[[example_index]], reference, read_codebook(guide))
    with (report / "local_sensitivity_example.json").open("w", encoding="utf-8") as file:
        json.dump(local, file, indent=2, allow_nan=False, default=lambda x: x.item())
    top = table.head(4)
    findings = f"""# Phase 10 — Explainability

Frozen model: `{selected['selected_model']}`. Global permutation importance was computed on a reproducible, class-stratified sample of 2,500 records from the 2020 validation year. It measures the decrease in macro F1 after shuffling each raw input, over three repeats. Rankings can depend on correlated features and this sample; a low value does not prove irrelevance.

## Highest validation-sample permutation effects

| Raw input | Macro F1 decrease | Repeat standard deviation |
| --- | ---: | ---: |
""" + "\n".join(f"| {r.feature} | {r.macro_f1_drop_mean:.4f} | {r.macro_f1_drop_std:.4f} |" for r in top.itertuples()) + f"""

![Permutation importance](permutation_importance.png)

The local example is in `local_sensitivity_example.json`. It replaces one raw feature at a time with a fixed reference (typical 2019 category/speed; noon and July 1 for temporal fields), then records the change in predicted-class probability. API explanations use the same method.

These are model-behavior descriptions, not causal effects. One-at-a-time substitutions may form unrealistic combinations; feature interactions are not fully allocated. The weighted Random Forest's class probabilities are not calibrated confidence. SHAP was not added: raw-feature permutation plus a small, auditable local sensitivity calculation avoids introducing another dependency and makes the comparison baseline explicit.
"""
    (report / "phase10_findings.md").write_text(findings, encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
