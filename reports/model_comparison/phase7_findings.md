# AccidentIQ — Phase 7: final model comparison

## Selection rule

The candidate was frozen before reading 2021 labels. Selection uses only mean macro F1 across the same three 2019 expanding-window folds. 2020 validation and 2021 test scores do not select the model. The selection metric is not a deployment-safety threshold.

| Model | 2019 CV macro F1 mean ± std |
| --- | ---: |
| random_forest_balanced | 0.3701 ± 0.0073 |
| lightgbm_balanced | 0.3509 ± 0.0082 |
| logistic_balanced | 0.3244 ± 0.0192 |
| pytorch_mlp | 0.3202 ± 0.0271 |
| lightgbm | 0.2945 ± 0.0021 |
| dummy_prior | 0.2936 ± 0.0021 |
| logistic | 0.2936 ± 0.0021 |

Selected model: **random_forest_balanced**.

## 2020 validation: class performance and operating cost

| Model | Accuracy | Macro F1 | Balanced accuracy | Fatal recall | Serious recall | Log-loss | 128-row latency (ms) | Artifact (KB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| random_forest_balanced | 0.6000 | 0.3667 | 0.4599 | 0.4206 | 0.2713 | 0.8911 | 107.01 | 9858.9 |
| lightgbm_balanced | 0.5287 | 0.3451 | 0.4647 | 0.4881 | 0.3240 | 0.9386 | 71.73 | 1293.2 |
| logistic_balanced | 0.4747 | 0.3193 | 0.4721 | 0.6175 | 0.2758 | 1.0208 | 73.27 | 11.4 |
| pytorch_mlp | 0.5547 | 0.3337 | 0.4733 | 0.5945 | 0.1734 | 0.9670 | 60.45 | 72.6 |
| lightgbm | 0.7835 | 0.2929 | 0.3334 | 0.0000 | 0.0001 | 0.5601 | 93.92 | 1288.5 |
| dummy_prior | 0.7835 | 0.2929 | 0.3333 | 0.0000 | 0.0000 | 0.5778 | 74.55 | 9.5 |
| logistic | 0.7835 | 0.2929 | 0.3333 | 0.0000 | 0.0000 | 0.5611 | 65.60 | 11.4 |

Latency is the median of five CPU `predict_proba` runs on 128 raw rows, including preprocessing. It is a local benchmark, not a service-level guarantee. Artifact sizes and class-specific precision/F1/AP/AUC are in `summary.json` and source phase reports. Train-vs-validation gaps, rare-class trade-offs, model size and latency should be reviewed together. Class weights increase minority influence but can lower precision and probability calibration.

## 2021 selected-model evaluation

| Metric | Selected model on 2021 |
| --- | ---: |
| Accuracy | 0.6024 |
| Macro F1 | 0.3679 |
| Balanced accuracy | 0.4594 |
| Fatal recall | 0.4152 |
| Serious recall | 0.2657 |
| Log-loss | 0.8874 |

The 2021 result is exploratory, not pristine: its yearly label distribution had already been inspected in Phase 2. During report generation, the final serialization step failed after scoring and was retried, so the identical frozen model was evaluated twice. No model, threshold or selection changed between runs; test results did not influence selection. Full per-class metrics and the repeated-evaluation note are in `summary.json`.

## Conclusion and limitations

The winning architecture is **random_forest_balanced** by the predeclared CV criterion. This dataset supports injury severity **conditional on a police-reported injury collision**. It does not contain non-collision road/time observations or exposure denominators, so it cannot estimate accident occurrence risk. The model is a research baseline, not an operational safety system. Multimodal methods were assessed separately and not trained because no reliable paired image events exist.

WHAT WE DID: compared Dummy, Logistic Regression, Random Forest, LightGBM and a PyTorch MLP under consistent chronological partitions.
WHY: to measure generalization, minority-class behavior, cost and model complexity before serving a model.
WHAT TO LEARN: macro F1, recall and probability metrics answer different questions; a class-weighted model can improve recall while producing more false positives. Cross-validation selects models, validation checks temporal generalization, and the final holdout is only a one-time estimate.
